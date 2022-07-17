import dataclasses
import json
import os
import re
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union
from io import StringIO
import numpy as np
import pandas as pd
import pickle

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import logging

logger = logging.getLogger(__name__)

ModelParameter = namedtuple("ModelParameter", ["name", "func_name", "init_value"])
gain = ModelParameter(
    name="gain", func_name="gainFunc", init_value=2.0
)  # gain + loss always
loss = ModelParameter(name="loss", func_name="lossFunc", init_value=2.0)
dupl = ModelParameter(
    name="dupl", func_name="duplFunc", init_value=2.0
)  # in all models except one: the base in gain+loss
demi_dupl = ModelParameter(name="demiPloidyR", func_name="demiDuplFunc", init_value=1.3)
base_num = ModelParameter(name="baseNum", func_name=None, init_value=3)
base_num_r = ModelParameter(name="baseNumR", func_name="baseNumRFunc", init_value=4.0)
model_parameters = [gain, loss, dupl, demi_dupl, base_num, base_num_r]
models = [
    [gain, loss],
    [gain, loss, dupl],
    [gain, loss, base_num, base_num_r],
    [gain, loss, dupl, demi_dupl],
    [gain, loss, dupl, base_num, base_num_r],
    [gain, loss, dupl, base_num, base_num_r, demi_dupl],
]
most_complex_model = "_".join(
    param.name for param in [gain, loss, dupl, base_num, base_num_r, demi_dupl]
)

included_parameter_template_with_func = (
    "\n_{param_name}_1 = {param_index};{param_init_value}\n_{func_name} = CONST\n"
)
included_parameter_template = "\n_{param_name}_1 = {param_index};{param_init_value}\n"
excluded_parameter_template = "\n_{func_name} = IGNORE\n"

states_frequencies_template = "\n_fixedFrequenciesFilePath = {frequencies_path}"
max_transitions_num_template = "\n_maxBaseNumTransition = {max_transitions_num}"
max_chr_inferred_template = "\n_maxChrInferred = {max_chr_inferred}"


@dataclass
class ChromevolInput:
    tree_path: str
    counts_path: str
    output_dir: str
    input_path: str
    parameters: Dict[str, float]  # map of parameter name to its initial value
    optimize_points_num: Union[str, int] = "10,3,1"
    optimize_iter_num: Union[str, int] = "0,2,5"
    min_chromosome_num: int = -1
    max_chromosome_num: int = -10
    num_trials: Union[str, int] = 10
    run_stochastic_mapping: bool = False
    num_of_simulations: int = 1000
    simulate: bool = False
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
        with open(
            f"{os.path.dirname(__file__)}/chromevol_param.template", "r"
        ) as infile:
            input_template = infile.read()
        input_string = input_template.format(**dataclasses.asdict(chromevol_input))
        parameters = input_args[
            "parameters"
        ]  # ASK TAL HOW TO DO THIS BETTER - I DON'T LIKE THIS
        param_index = 1
        for param in model_parameters:
            if param.name in parameters:
                if param.func_name is not None:
                    input_string += included_parameter_template_with_func.format(
                        param_name=param.name,
                        param_index=param_index,
                        param_init_value=input_args["parameters"][param.name],
                        func_name=param.func_name,
                    )
                else:
                    input_string += included_parameter_template.format(
                        param_name=param.name,
                        param_index=param_index,
                        param_init_value=input_args["parameters"][param.name],
                    )
                param_index += 1
            elif param.func_name is not None:
                input_string += excluded_parameter_template.format(
                    func_name=param.func_name
                )
        if input_args.get("frequencies_path", None) is not None:
            input_string += states_frequencies_template.format(
                frequencies_path=input_args["frequencies_path"]
            )
        if input_args.get("max_transitions_num", None) is not None:
            input_string += max_transitions_num_template.format(
                max_transitions_num=input_args["max_transitions_num"]
            )
        if input_args.get("max_chr_inferred", None) is not None:
            input_string += max_chr_inferred_template.format(
                max_chr_inferred=input_args["max_chr_inferred"]
            )

        input_string = input_string.replace("False", "false").replace(
            "True", "true"
        )  # ASK TAL HOW TO DO THIS BETTER - I DON'T LIKE THIS
        input_string = input_string.replace("//", "/")
        with open(chromevol_input.input_path, "w") as outfile:
            outfile.write(input_string)
        return chromevol_input

    @staticmethod
    def _exec(chromevol_input_path: str) -> int:
        cmd = f"{os.getenv('CONDA_ACT_CMD')};cd {os.path.dirname(chromevol_input_path)};{os.getenv('CHROMEVOL_EXEC')} param={os.path.abspath(chromevol_input_path)}"
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
        parameters_regex = re.compile(
            "Final model parameters are\:(.*?)AIC", re.MULTILINE | re.DOTALL
        )
        try:
            parameters_str = parameters_regex.search(result_str).group(1)
            parameter_regex = re.compile("Chromosome\.(.*?)0*_1\s=\s(\d*\.*\d*)")
            parameters = dict()
            for match in parameter_regex.finditer(parameters_str):
                parameters[match.group(1)] = float(match.group(2))
            return parameters
        except Exception as e:
            logger.error(
                f"failed to extract model parameters from {result_str} due to error {e}"
            )
            return np.nan

    @staticmethod
    def _get_root_chromosome_number(result_str: str) -> int:
        root_chromosome_number_regex = re.compile(
            "Ancestral chromosome number at the root\:\s(\d*)"
        )
        try:
            return int(root_chromosome_number_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(
                f"failed to extract root chromosome number from {result_str} due to error {e}"
            )
            return np.nan

    @staticmethod
    def _get_log_likelihood(result_str: str) -> float:
        log_likelihood_regex = re.compile(
            "Final optimized likelihood is\:\s(-*\d*\.?\d*)"
        )
        try:
            return float(log_likelihood_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(
                f"failed to extract log likelihood from {result_str} due to error {e}"
            )
            return np.nan

    @staticmethod
    def _get_model_score(result_str: str) -> float:
        aicc_score_regex = re.compile("AICc? of the best model\s=\s(\d*\.?\d*)")
        try:
            return float(aicc_score_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(
                f"failed to extract AICc score from {result_str} due to error {e}"
            )
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
        scaling_factor_regex = re.compile("Tree scaling factor is\:\s(\d*\.?\d*)")
        try:
            return float(scaling_factor_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(
                f"failed to extract AICc score from {result_str} due to error {e}"
            )
            return np.nan

    @staticmethod
    def _parse_result(
        path: str,
    ) -> Tuple[float, float, int, Dict[str, float], float, str]:
        with open(path, "r") as infile:
            result_str = infile.read()
        maximum_likelihood_estimators = ChromevolExecutor._get_model_parameters(
            result_str=result_str
        )
        root_chromosome_number = ChromevolExecutor._get_root_chromosome_number(
            result_str=result_str
        )
        log_likelihood = ChromevolExecutor._get_log_likelihood(result_str=result_str)
        model_score = ChromevolExecutor._get_model_score(result_str=result_str)
        tree_scaling_factor = ChromevolExecutor._get_tree_scaling_factor(
            result_str=result_str
        )
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
                logger.error(
                    f"failed to extract expectations data from {input_path} due to error {e}"
                )
        else:
            logger.info(f"expectations computation was not conduced in this execution")

    @staticmethod
    def _parse_stochastic_mappings(input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for path in os.listdir(input_dir):
            if path.endswith(".csv"):
                os.rename(f"{input_dir}/{path}", f"{output_dir}/{path}")

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
        ChromevolExecutor._parse_stochastic_mappings(
            input_dir=output_dir, output_dir=stochastic_mappings_dir
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
        raw_output_path = f"{chromevol_input.output_dir}/chromEvol.res"
        res = ChromevolExecutor._exec(chromevol_input_path=chromevol_input.input_path)
        chromevol_output = None
        if res != 0:
            res = ChromevolExecutor._retry(input_args=input_args)
            if res != 0:
                logger.warning(
                    f"retry failed execute chromevol on {chromevol_input.input_path}"
                )
                return chromevol_output
        if os.path.exists(raw_output_path):
            chromevol_output = ChromevolExecutor._parse_output(
                chromevol_input.output_dir
            )
        return chromevol_output
