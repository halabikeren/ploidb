import dataclasses
import json
import os
import re
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
from io import StringIO
import numpy as np
import pandas as pd
import pickle

from dotenv import load_dotenv, find_dotenv
from ete3 import Tree

load_dotenv(find_dotenv())

import logging

logger = logging.getLogger(__name__)

ModelParameter = namedtuple("ModelParameter", ["input_name", "output_name", "func_name", "init_value"])
gain = ModelParameter(input_name="gain", output_name="gain", func_name="gainFunc", init_value=2.0)  # gain + loss always
loss = ModelParameter(input_name="loss", output_name="loss", func_name="lossFunc", init_value=2.0)
dupl = ModelParameter(
    input_name="dupl", output_name="dupl", func_name="duplFunc", init_value=2.0
)  # in all models except one: the base in gain+loss
demi_dupl = ModelParameter(input_name="demiPloidyR", output_name="demi", func_name="demiDuplFunc", init_value=1.3)
base_num = ModelParameter(input_name="baseNum", output_name="baseNum", func_name=None, init_value=6)
base_num_r = ModelParameter(input_name="baseNumR", output_name="baseNumR", func_name="baseNumRFunc", init_value=4.0)
model_parameters = [gain, loss, dupl, demi_dupl, base_num, base_num_r]
models_without_base_num = [
    [gain, loss],
    [gain, loss, dupl],
    [gain, loss, dupl, demi_dupl],
]
models_with_base_num = [
    [gain, loss, base_num, base_num_r],
    [gain, loss, dupl, base_num, base_num_r],
    [gain, loss, dupl, base_num, base_num_r, demi_dupl],
]

models = models_without_base_num + models_with_base_num
most_complex_model_with_base_num = "_".join(
    param.input_name for param in [gain, loss, dupl, base_num, base_num_r, demi_dupl]
)
most_complex_model_without_base_num = "_".join(param.input_name for param in [gain, loss, dupl, demi_dupl])
counts_path_template = "_dataFile = {counts_path}"
included_parameter_template_with_func = "\n_{param_name}_1 = {param_index};{param_init_value}\n_{func_name} = CONST\n"
included_parameter_template = "\n_{param_name}_1 = {param_index};{param_init_value}\n"
excluded_parameter_template = "\n_{func_name} = IGNORE\n"

states_frequencies_template = "\n_fixedFrequenciesFilePath = {frequencies_path}"
max_transitions_num_template = "\n_maxBaseNumTransition = {max_transitions_num}"
max_chr_inferred_template = "\n_maxChrInferred = {max_chr_inferred}"


@dataclass
class ChromevolInput:
    tree_path: str
    output_dir: str
    input_path: str
    parameters: Dict[str, float]  # map of parameter name to its initial value
    counts_path: Optional[str] = None  # this argument in none is case of simulations
    optimize_points_num: Union[str, int] = "10,3,1"
    optimize_iter_num: Union[str, int] = "0,2,5"
    min_chromosome_num: int = -1
    max_chromosome_num: int = -10
    num_trials: Union[str, int] = 10
    run_stochastic_mapping: bool = False
    num_of_mappings: int = 1000
    simulate: bool = False
    num_of_simulation_trials: int = 1000
    num_of_simulations: int = 100
    allowed_failed_sim_frac: float = 0.9
    tree_scaling_factor: float = 999
    states_frequencies_path: Optional[str] = None
    max_transitions_num: Optional[int] = None
    max_chr_inferred: Optional[int] = None
    frequencies_path: Optional[str] = None  # input in is a text format
    seed: int = 1
    model_selection_criterion: str = "AIC"  # other option: AICc


@dataclass
class ChromevolOutput:
    result_path: str
    expected_events_path: str
    stochastic_mappings_dir: str
    root_chromosome_number: int
    log_likelihood: float
    model_score: float  # could be AIC or AICc
    model_parameters: Dict[str, float]
    tree_scaling_factor: Optional[float] = None
    states_frequencies_path: Optional[str] = None  # output is in a pickle format


class ChromevolExecutor:
    @staticmethod
    def _get_input(input_args: Dict[str, str]) -> ChromevolInput:
        chromevol_input = ChromevolInput(**input_args)
        with open(f"{os.path.dirname(__file__)}/chromevol_param.template", "r") as infile:
            input_template = infile.read()
        input_string = input_template.format(**dataclasses.asdict(chromevol_input))
        parameters = input_args["parameters"]  # ASK TAL HOW TO DO THIS BETTER - I DON'T LIKE THIS
        param_index = 1
        for param in model_parameters:
            if param.input_name in parameters:
                if param.func_name is not None:
                    input_string += included_parameter_template_with_func.format(
                        param_name=param.input_name,
                        param_index=param_index,
                        param_init_value=input_args["parameters"][param.input_name],
                        func_name=param.func_name,
                    )
                else:
                    input_string += included_parameter_template.format(
                        param_name=param.input_name,
                        param_index=param_index,
                        param_init_value=input_args["parameters"][param.input_name],
                    )
                param_index += 1
            elif param.func_name is not None:
                input_string += excluded_parameter_template.format(func_name=param.func_name)
        if input_args.get("counts_path", None) is not None:
            input_string += counts_path_template.format(counts_path=input_args["counts_path"])
        if input_args.get("frequencies_path", None) is not None:
            input_string += states_frequencies_template.format(frequencies_path=input_args["frequencies_path"])
        if input_args.get("max_transitions_num", None) is not None:
            input_string += max_transitions_num_template.format(max_transitions_num=input_args["max_transitions_num"])
        if input_args.get("max_chr_inferred", None) is not None:
            input_string += max_chr_inferred_template.format(max_chr_inferred=input_args["max_chr_inferred"])

        input_string = input_string.replace("False", "false").replace(
            "True", "true"
        )  # ASK TAL HOW TO DO THIS BETTER - I DON'T LIKE THIS
        input_string = input_string.replace("//", "/")
        with open(chromevol_input.input_path, "w") as outfile:
            outfile.write(input_string)
        return chromevol_input

    @staticmethod
    def _exec(chromevol_input_path: str) -> int:
        cmd = f"{os.getenv('CONDA_ACT_CMD')};cd {os.path.dirname(chromevol_input_path)};{os.getenv('CHROMEVOL_EXEC')} param={os.path.abspath(chromevol_input_path)} > {os.path.dirname(os.path.abspath(chromevol_input_path))}/chromevol.log"
        cmd.replace("//", "/")
        res = os.system(cmd)
        return res

    @staticmethod
    def _retry(input_args: Dict[str, str]) -> int:
        input_args["num_trials"] = "15"
        input_args["optimize_points_num"] = "2,1"
        input_args["optimize_iter_num"] = "2,5"
        chromevol_input = ChromevolExecutor._get_input(input_args=input_args)
        res = ChromevolExecutor._exec(chromevol_input_path=chromevol_input.input_path)
        return res

    @staticmethod
    def _get_model_parameters(result_str: str) -> Dict[str, float]:
        parameters_regex = re.compile("Final model parameters are\:(.*?)AIC", re.MULTILINE | re.DOTALL)
        try:
            parameters_str = parameters_regex.search(result_str).group(1)
            parameter_regex = re.compile("Chromosome\.(.*?)0*_1\s=\s(\d*\.*\d*e*-*\d.*)")
            parameters = dict()
            param_out_to_in_name = {param.output_name: param.input_name for param in model_parameters}
            for match in parameter_regex.finditer(parameters_str):
                param_out_name = match.group(1)
                param_mle = float(match.group(2))
                param_in_name = param_out_to_in_name[param_out_name]
                parameters[param_in_name] = param_mle
            return parameters
        except Exception as e:
            logger.error(f"failed to extract model parameters from {result_str} due to error {e}")
            return np.nan

    @staticmethod
    def _get_root_chromosome_number(result_str: str) -> int:
        root_chromosome_number_regex = re.compile("Ancestral chromosome number at the root\:\s(\d*)")
        try:
            return int(root_chromosome_number_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(f"failed to extract root chromosome number from {result_str} due to error {e}")
            return np.nan

    @staticmethod
    def _get_log_likelihood(result_str: str) -> float:
        log_likelihood_regex = re.compile("Final optimized likelihood is\:\s(-*\d*\.?\d*)")
        try:
            return float(log_likelihood_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(f"failed to extract log likelihood from {result_str} due to error {e}")
            return np.nan

    @staticmethod
    def _get_model_score(result_str: str) -> float:
        aicc_score_regex = re.compile("AICc? of the best model\s=\s(\d*\.?\d*)")
        try:
            return float(aicc_score_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(f"failed to extract AICc score from {result_str} due to error {e}")
            return np.nan

    @staticmethod
    def _get_states_frequencies(result_str: str) -> dict[int, float]:
        state_to_freq_regex = re.compile("F\[(\d*)\]\s=\s(\d*\.?\d*[e-]*\d*)")
        state_to_freq = dict()
        for match in state_to_freq_regex.finditer(result_str):
            state_to_freq[int(match.group(1))] = float(match.group(2))
        return state_to_freq

    @staticmethod
    def _get_tree_scaling_factor(result_str: str) -> float:
        scaling_factor_regex = re.compile("Tree scaling factor is\:\s(\d*\.?\d*[e-]*\d*)")
        try:
            return float(scaling_factor_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(f"failed to extract AICc score from {result_str} due to error {e}")
            return np.nan

    @staticmethod
    def _parse_result(
        path: str,
    ) -> Tuple[float, float, int, Dict[str, float], float, str]:
        with open(path, "r") as infile:
            result_str = infile.read()
        maximum_likelihood_estimators = ChromevolExecutor._get_model_parameters(result_str=result_str)
        root_chromosome_number = ChromevolExecutor._get_root_chromosome_number(result_str=result_str)
        log_likelihood = ChromevolExecutor._get_log_likelihood(result_str=result_str)
        model_score = ChromevolExecutor._get_model_score(result_str=result_str)
        tree_scaling_factor = ChromevolExecutor._get_tree_scaling_factor(result_str=result_str)
        state_to_freq = ChromevolExecutor._get_states_frequencies(result_str=result_str)
        states_frequencies_path = f"{os.path.dirname(path)}/root_frequencies.pkl"
        with open(states_frequencies_path, "wb") as outfile:
            pickle.dump(obj=state_to_freq, file=outfile)
        return (
            log_likelihood,
            model_score,
            root_chromosome_number,
            maximum_likelihood_estimators,
            tree_scaling_factor,
            states_frequencies_path,
        )

    @staticmethod
    def _parse_expectations(input_path: str, output_path: str):
        if os.path.exists(input_path):
            with open(input_path, "r") as infile:
                expectations = infile.read()
            expectations_regex = re.compile(
                "#EXPECTED NUMBER OF EVENTS FROM ROOT TO LEAF(.*?)\#",
                re.MULTILINE | re.DOTALL,
            )
            try:
                expectations_str = expectations_regex.search(expectations).group(1)
                expectations_data = pd.read_csv(StringIO(expectations_str), sep="\t")
                expectations_data.to_csv(output_path, index=False)
            except Exception as e:
                logger.error(f"failed to extract expectations data from {input_path} due to error {e}")
        else:
            logger.info(f"expectations computation was not conduced in this execution")

    @staticmethod
    def _get_event_type(src_state: int, dst_state: int) -> str:
        if src_state + 1 == dst_state:
            return "GAIN"
        elif src_state - 1 == dst_state:
            return "LOSS"
        elif src_state * 2 == dst_state:
            return "DUPLICATION"
        elif src_state * 1.5 == dst_state:
            return "DEMI-DUPLICATION"
        elif dst_state == 200:
            return "TOMAX"
        return "BASE-NUMBER"

    @staticmethod
    def parse_ml_tree(tree_path: str, scaling_factor: float) -> Tree:
        tree = Tree(tree_path, format=1)
        for node in tree.traverse():
            node.name = "-".join(node.name.split("-")[:-1])
            node.dist = node.dist / scaling_factor
        return tree

    @staticmethod
    def parse_evolutionary_path(input_path: str, output_path: str, tree: Tree, tree_scaling_factor: float) -> int:
        branch_id_regex = re.compile("(.*?)\nFather is\: (.*?)\n")
        transition_regex = re.compile("from state\:\s*(\d*)\s*t\s*=\s*(\d*\.?\d*)\s*to\s*state\s*=\s*(\d*)")
        node_to_is_external = {node.name: node.is_leaf() for node in tree.traverse()}
        tree_length = tree.get_distance(tree.get_leaves()[0])
        node_is_legal = {node.name: node.dist < (tree_length / 2) for node in tree.traverse()}
        event_ages_dfs = []
        with open(input_path, "r") as f:
            history_paths = f.read().split("*************************************")
        for path in history_paths:
            branch_id_match = branch_id_regex.search(path)
            if branch_id_match is None:
                continue
            branch_child = branch_id_match.group(1).replace("N-", "N")
            branch_parent = branch_id_match.group(2).replace("N-", "N")
            child = tree.search_nodes(name=branch_child)[0]
            parent = tree.search_nodes(name=branch_parent)[0]
            base_age = parent.get_distance(parent.get_leaves()[0])
            curr_age = base_age
            min_age = child.get_distance(child.get_leaves()[0])
            assert np.round(child.dist, 0) == np.round(base_age - min_age, 0)
            transitions = [match for match in transition_regex.finditer(path)]
            src_states = [int(match.group(1)) for match in transitions]
            dst_states = [int(match.group(3)) for match in transitions]
            time_to_transition = np.array([float(match.group(2)) / tree_scaling_factor for match in transitions])
            branch_length = base_age - min_age
            sum_of_transitions = np.sum(time_to_transition)
            if np.round(sum_of_transitions, 3) > np.round(branch_length, 3):
                logger.info(
                    f"the sum of transitions along branch ({parent.name}, {child.name}) adds up to more than {sum_of_transitions}, suggesting that the branch was stretched, and will thus be ignored"
                )
                continue
            for i in range(len(transitions)):
                src_state = src_states[i]
                curr_age -= time_to_transition[i]
                assert np.round(curr_age, 3) >= np.round(min_age, 3)
                dst_state = dst_states[i]
                event_type = ChromevolExecutor._get_event_type(src_state, dst_state)
                event_ages_dfs.append(
                    pd.DataFrame.from_dict(
                        {
                            "age": curr_age,
                            "branch_parent_name": branch_parent,
                            "branch_child_name": branch_child,
                            "event_type": event_type,
                            "src_state": src_state,
                            "dst_state": dst_state,
                            "is_child_external": node_to_is_external.get(branch_child, np.nan),
                            "is_legal": node_is_legal.get(branch_child, np.nan),
                        },
                        orient="index",
                    ).transpose()
                )

        if len(event_ages_dfs) > 0:
            event_ages = pd.concat(event_ages_dfs)
            event_ages.to_csv(output_path, index=False)
        return 0

    @staticmethod
    def _parse_simulations(input_dir: str, evolutionary_paths_dir: str):
        pass

    @staticmethod
    def _parse_stochastic_mappings(input_dir: str, mappings_dir: str, evolutionary_paths_dir: str):
        os.makedirs(mappings_dir, exist_ok=True)
        for path in os.listdir(input_dir):
            if path.endswith(".csv"):
                os.rename(f"{input_dir}/{path}", f"{mappings_dir}/{path}")
        res = os.system(
            f"cd {input_dir};zip -r stochastic_mappings.zip stochastic_mappings/;rm -rf ./stochastic_mappings/"
        )

        os.makedirs(evolutionary_paths_dir, exist_ok=True)
        raw_evolutionary_paths_dir = f"{input_dir}raw_evolutionary_paths/"
        os.makedirs(raw_evolutionary_paths_dir, exist_ok=True)

        tree_path = f"{input_dir}MLAncestralReconstruction.tree"
        with open(f"{input_dir}sm_params.json", "r") as f:
            scaling_factor = float(json.load(f)["tree_scaling_factor"])
        tree = ChromevolExecutor.parse_ml_tree(tree_path=tree_path, scaling_factor=scaling_factor)

        for path in os.listdir(input_dir):
            if path.startswith("evoPathMapping_"):
                index = path.replace("evoPathMapping_", "").replace(".txt", "")
                input_path = f"{input_dir}{path}"
                output_path = f"{evolutionary_paths_dir}events_by_age_simulations_{index}.csv"
                res = ChromevolExecutor.parse_evolutionary_path(
                    input_path=input_path, output_path=output_path, tree=tree, tree_scaling_factor=scaling_factor
                )
                res = os.system(f"mv {input_path} {raw_evolutionary_paths_dir}")
        res = os.system(
            f"cd {input_dir};zip -r raw_evolutionary_paths.zip ./raw_evolutionary_paths; rm -rf ./raw_evolutionary_paths/"
        )
        res = os.system(
            f"cd {input_dir};zip -r evolutionary_paths.zip ./evolutionary_paths; rm -rf ./evolutionary_paths/"
        )

    @staticmethod
    def _parse_output(output_dir: str) -> ChromevolOutput:
        result_path = f"{output_dir}/chromEvol.res"
        expected_events_path = f"{output_dir}/expectations_root_to_tip.csv"
        (
            log_likelihood,
            model_score,
            root_chromosome_number,
            mles,
            tree_scaling_factor,
            states_frequencies_path,
        ) = ChromevolExecutor._parse_result(path=result_path)
        ChromevolExecutor._parse_expectations(
            input_path=f"{output_dir}/expectations_second_round.txt",
            output_path=expected_events_path,
        )
        stochastic_mappings_dir = f"{output_dir}/stochastic_mappings/"
        sm_input_path = f"{output_dir}/sm_params.json"
        if os.path.exists(sm_input_path):
            stochastic_evolutionary_paths_dir = f"{output_dir}/evolutionary_paths/"
            ChromevolExecutor._parse_stochastic_mappings(
                input_dir=output_dir,
                mappings_dir=stochastic_mappings_dir,
                evolutionary_paths_dir=stochastic_evolutionary_paths_dir,
            )
        chromevol_output = ChromevolOutput(
            result_path=result_path,
            expected_events_path=expected_events_path,
            stochastic_mappings_dir=stochastic_mappings_dir,
            root_chromosome_number=root_chromosome_number,
            log_likelihood=log_likelihood,
            model_score=model_score,
            model_parameters=mles,
            tree_scaling_factor=tree_scaling_factor,
            states_frequencies_path=states_frequencies_path,
        )
        with open(f"{output_dir}/parsed_output.json", "w") as outfile:
            json.dump(obj=dataclasses.asdict(chromevol_output), fp=outfile)
        return chromevol_output

    @staticmethod
    def run(input_args: Dict[str, str]) -> ChromevolOutput:
        chromevol_input = ChromevolExecutor._get_input(input_args=input_args)
        if "base_num" in chromevol_input.parameters:
            chromevol_input.parameters["base_num"] = max(chromevol_input.parameters["base_num"], 6)
        raw_output_path = f"{chromevol_input.output_dir}/chromEvol.res"
        if not os.path.exists(raw_output_path):
            res = ChromevolExecutor._exec(chromevol_input_path=chromevol_input.input_path)
            chromevol_output = None
            if res != 0:
                res = ChromevolExecutor._retry(input_args=input_args)
                if res != 0:
                    logger.warning(f"retry failed execute chromevol on {chromevol_input.input_path}")
                    return chromevol_output
        if os.path.exists(raw_output_path):
            chromevol_output = ChromevolExecutor._parse_output(chromevol_input.output_dir)
        return chromevol_output
