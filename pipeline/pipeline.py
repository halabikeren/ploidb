import json
import os
import re
import shutil
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import sys

import pickle
from Bio import SeqIO, Phylo
from ete3 import Tree

from timeit import default_timer as timer
from datetime import timedelta

from scipy import optimize
from sklearn.metrics import matthews_corrcoef

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_generation.chromevol import (
    model_parameters,
    models,
    most_complex_model,
    ChromevolOutput,
)
from services.pbs_service import PBSService

import logging

logger = logging.getLogger(__name__)


class Pipeline:
    work_dir: str
    parallel: bool
    ram_per_job: int
    queue: str
    max_parallel_jobs: int

    def __init__(self,
                 work_dir: str = f"{os.getcwd()}/ploidb_pipeline/",
                 parallel: bool = False,
                 ram_per_job: int = 1,
                 queue: str = "itaym",
                 max_parallel_jobs: int = 1000,
                ):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.parallel = parallel
        self.ram_per_job = ram_per_job
        self.queue = queue
        self.max_parallel_jobs = max_parallel_jobs

    def _send_chromevol_commands(
            self,
            command_dir: str,
            input_to_output: dict[str, str],
            input_to_run_in_main: Optional[str] = None,
    ) -> int:
        os.makedirs(command_dir, exist_ok=True)
        input_to_jobs_commands = dict()
        for input_path in input_to_output:
            if not os.path.exists(input_to_output[input_path]):
                if input_to_run_in_main and input_path == input_to_run_in_main:
                    continue
                input_to_jobs_commands[input_path] = [
                    os.getenv("CONDA_ACT_CMD"),
                    f"cd {os.path.dirname(input_path)}",
                    f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={input_path}",
                ]
        main_cmd = None
        if input_to_run_in_main and not os.path.exists(input_to_output[input_to_run_in_main]):
                main_cmd = f"{os.getenv('CONDA_ACT_CMD')}; cd {command_dir};python {os.path.dirname(__file__)}/run_chromevol.py --input_path={input_to_run_in_main}"

        logger.info(
            f"# chromevol jobs to run = {len(input_to_jobs_commands.keys()) + 1 if main_cmd is not None else 0}"
        )
        if len(input_to_jobs_commands.keys()) > 0:
            if self.parallel:
                jobs_paths = PBSService.generate_jobs(
                    jobs_commands=[input_to_jobs_commands[input_path] for input_path in input_to_jobs_commands if input_path != input_to_run_in_main],
                    work_dir=f"{command_dir}jobs/",
                    output_dir=f"{command_dir}jobs_output/",
                    ram_per_job_gb=self.ram_per_job,
                    queue=self.queue,
                )
                jobs_ids = PBSService.submit_jobs(
                    jobs_paths=jobs_paths, max_parallel_jobs=self.max_parallel_jobs, queue=self.queue,
                )  # make sure to always submit these jobs
                done = False
                while not done:
                    PBSService.wait_for_jobs(jobs_ids=jobs_ids)
                    job_ids = PBSService.retry_memory_failures(jobs_paths=jobs_paths,
                                                               jobs_output_dir=f"{command_dir}jobs_output/")
                    done = (len(job_ids) == 0)
            else:
                for input_path in input_to_jobs_commands:
                    job_commands_set = input_to_jobs_commands[input_path]
                    logger.info(f"submitting chromevol process for {input_path}")
                    res = os.system(";".join(job_commands_set))
                    if not os.path.exists(input_to_output[input_path]):
                        logger.error(
                            f"execution of {input_path} fit failed after retry"
                        )
            # in any case, run the main command from the parent process

        if main_cmd is not None:
            res = os.system(main_cmd)

        logger.info(
            f"completed execution of chromevol across {len(input_to_jobs_commands.keys())} inputs"
        )
        return 0

    @staticmethod
    def _create_models_input_files(
        tree_path: str, counts_path: str, work_dir: str
    ) -> dict:
        model_to_io = dict()
        base_input_ags = {
            "tree_path": tree_path,
            "counts_path": counts_path,
            "num_of_simulations": 0,
            "run_stochastic_mapping": False,
        }
        for model in models:
            model_name = "_".join([param.input_name for param in model])
            model_dir = f"{os.path.abspath(work_dir)}/{model_name}/"
            os.makedirs(model_dir, exist_ok=True)
            model_input_args = base_input_ags.copy()
            model_input_args["input_path"] = f"{model_dir}input.params"
            model_input_args["output_dir"] = model_dir
            model_input_args["parameters"] = {
                param.input_name: param.init_value for param in model
            }
            model_input_path = f"{model_dir}input.json"
            with open(model_input_path, "w") as infile:
                json.dump(obj=model_input_args, fp=infile)
            model_to_io[model_name] = {
                "input_path": model_input_path,
                "output_path": f"{model_dir}/parsed_output.json",
            }
        logger.info(
            f"created {len(models)} model input files for the considered models across the parameter set {','.join([param.input_name for param in model_parameters])}"
        )
        return model_to_io

    @staticmethod
    def _select_best_model(model_to_io: Dict[str, Dict[str, str]]) -> str:
        winning_model = np.nan
        winning_model_score = float("inf")
        for model_name in model_to_io:
            model_output_path = model_to_io[model_name]["output_path"]
            with open(model_output_path, "r") as outfile:
                model_output = ChromevolOutput(**json.load(fp=outfile))
            model_score = model_output.model_score
            if model_score < winning_model_score:
                winning_model, winning_model_score = model_output_path, model_score
        logger.info(
            f"the selected model is {winning_model} with a score of {winning_model_score}"
        )
        if pd.isna(winning_model):
            raise ValueError(f"failed to train any model on the dataset")
        return winning_model

    def get_best_model(
            self,
            counts_path: str,
            tree_path: str,
    ) -> str:
        model_selection_work_dir = f"{self.work_dir}/model_selection/"
        os.makedirs(model_selection_work_dir, exist_ok=True)
        model_to_io = self._create_models_input_files(
            tree_path=tree_path, counts_path=counts_path, work_dir=model_selection_work_dir,
        )
        input_to_output = {
            model_to_io[model_name]["input_path"]: model_to_io[model_name]["output_path"]
            for model_name in model_to_io
        }
        most_complex_model_input_path = model_to_io[most_complex_model]["input_path"]
        res = os.system(f"rm -rf {model_selection_work_dir}/jobs/")
        res = os.system(f"rm -rf {model_selection_work_dir}/jobs_output/")
        res = self._send_chromevol_commands(
            command_dir=model_selection_work_dir,
            input_to_output=input_to_output,
            input_to_run_in_main=most_complex_model_input_path,
        )
        models_to_del = []
        for model_name in model_to_io:
            output_path = model_to_io[model_name]["output_path"]
            if not os.path.exists(output_path):
                logger.error(
                    f"execution of model {model_name} fit failed after retry and thus the model will be excluded from model selection"
                )
                models_to_del.append(model_name)
        for model_name in models_to_del:
            del model_to_io[model_name]
        return self._select_best_model(model_to_io)

    @staticmethod
    def _create_stochastic_mapping_input(
        counts_path: str,
        tree_path: str,
        model_parameters_path: str,
        work_dir: str,
        mappings_num: int,
        optimize_model: bool = False,
    ) -> Tuple[str, str]:
        sm_input_path = f"{work_dir}sm_params.json"
        chromevol_input_path = f"{work_dir}sm.params"
        sm_output_dir = f"{work_dir}stochastic_mappings/"
        input_args = {
            "input_path": chromevol_input_path,
            "tree_path": tree_path,
            "counts_path": counts_path,
            "output_dir": work_dir,
            "run_stochastic_mapping": True,
            "num_of_mappings": mappings_num,
        }
        if not optimize_model:
            input_args["optimize_points_num"] = 1
            input_args["optimize_iter_num"] = 0
        with open(model_parameters_path, "r") as infile:
            selected_model_res = json.load(fp=infile)
            input_args["parameters"] = selected_model_res["model_parameters"]
            input_args["tree_scaling_factor"] = selected_model_res[
                "tree_scaling_factor"
            ]
        with open(sm_input_path, "w") as outfile:
            json.dump(obj=input_args, fp=outfile)
        return sm_input_path, sm_output_dir

    def _get_stochastic_mappings(
        self,
        counts_path: str,
        tree_path: str,
        model_parameters_path: str,
        mappings_num: int = 100,
        optimize_model: bool = False,
    ) -> pd.DataFrame:
        sm_work_dir = self.work_dir
        if counts_path.startswith(f"{self.work_dir}/simulations/"):
            sm_work_dir = os.path.dirname(counts_path)
        sm_work_dir += "/stochastic_mapping/"
        os.makedirs(sm_work_dir, exist_ok=True)
        sm_input_path, sm_output_dir = self._create_stochastic_mapping_input(
            counts_path=counts_path,
            tree_path=tree_path,
            model_parameters_path=model_parameters_path,
            work_dir=sm_work_dir,
            mappings_num=mappings_num,
            optimize_model=optimize_model,
        )
        if (
            not os.path.exists(sm_output_dir)
            or len(os.listdir(sm_output_dir)) < mappings_num
        ):
            commands = [
                os.getenv("CONDA_ACT_CMD"),
                f"cd {sm_work_dir}",
                f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={sm_input_path}",
            ]
            res = os.system(";".join(commands)) # add parallelization
        logger.info(f"computed {mappings_num} mappings successfully")
        taxon_to_polyploidy_support = self._get_frequency_of_duplication_events(
            sm_work_dir=sm_work_dir,
            mappings_num=mappings_num,
        )
        return taxon_to_polyploidy_support

    @staticmethod
    def _get_true_values(
        mappings: List[pd.DataFrame], classification_field: str
    ) -> np.array:
        tip_taxa = mappings[0]["NODE"].to_list()
        true_values = pd.DataFrame(columns=["mapping"] + tip_taxa)
        true_values["mapping"] = list(range(len(mappings)))
        for taxon in tip_taxa:
            true_values[taxon] = true_values["mapping"].apply(
                lambda i: 1
                if mappings[i]
                .loc[mappings[i]["NODE"] == taxon, classification_field]
                .values[0]
                else -1
            )
        return true_values.set_index("mapping").to_numpy().flatten()

    @staticmethod
    def _get_predicted_values(
        ploidy_data: pd.DataFrame, for_polyploidy: bool, freq_threshold: float
    ) -> np.array:
        is_upper_bound = False if for_polyploidy else True
        predicted_values = (
            ploidy_data.duplication_events_frequency <= freq_threshold
            if is_upper_bound
            else ploidy_data.duplication_events_frequency >= freq_threshold
        )

        predicted_values.index = ploidy_data["node"]
        return predicted_values

    @staticmethod
    def _get_optimal_threshold(
        ploidy_data: pd.DataFrame, for_polyploidy: bool = True, upper_bound: float = 1,
    ) -> tuple[float, float, pd.DataFrame]:
        positive_label_code = 1 if for_polyploidy else 0
        true_values = (
            (ploidy_data["is_polyploid"] == positive_label_code)
            .astype(np.int16)
            .tolist()
        )
        num_sim = len(ploidy_data.simulation.unique())
        thresholds = [
            i * 0.1
            for i in range(4 if for_polyploidy else 0, 11)
            if i * 0.1 <= upper_bound
        ]
        logger.info(
            f"maximal threshold to examine  for {'polyploidy' if for_polyploidy else 'diploidy'} threshold = {thresholds[-1]} across {num_sim:,} simulations"
        )

        def target_func(x, args):
            predicted_values = Pipeline._get_predicted_values(
                ploidy_data=args["ploidy_data"],
                for_polyploidy=args["for_polyploidy"],
                freq_threshold=x[0],
            )
            coeff = np.round(
                matthews_corrcoef(
                    y_true=true_values,
                    y_pred=predicted_values.astype(np.int16).tolist(),
                ),
                len(str(num_sim)),
            )
            logger.info(f"thr={x[0]}, coeff={coeff}")
            args["memoization"][x[0]] = coeff
            return -1 * coeff  # maximize coeff

        lower_bound = 0.5 if for_polyploidy else 0
        bounds = ((lower_bound, upper_bound),)
        examined_thresholds = dict()
        minimizer_kwargs = {
            "method": "SLSQP",
            "args": {
                "ploidy_data": ploidy_data,
                "for_polyploidy": for_polyploidy,
                "true_values": true_values,
                "memoization": examined_thresholds,
            },
            "bounds": bounds,
            "tol": 1e-3,
        }
        poly_sp = np.nanmax(
            [
                0.5,
                ploidy_data.loc[
                    ploidy_data.is_polyploid == 1, "duplication_events_frequency"
                ].min(),
            ],
        )
        di_sp = np.nanmin(
            [
                ploidy_data.loc[
                    ploidy_data.is_polyploid == 0, "duplication_events_frequency"
                ].max(),
                upper_bound,
            ]
        )
        di_sp = di_sp if di_sp > 0 else  upper_bound/2
        logger.info(
            f"optimizing {'poly' if for_polyploidy else 'di'}ploidy threshold with a starting point of {poly_sp if for_polyploidy else di_sp}"
        )
        optimized_res = optimize.basinhopping(
            func=target_func,
            x0=[poly_sp] if for_polyploidy else [di_sp],
            minimizer_kwargs=minimizer_kwargs,
            stepsize=0.05,
            niter_success=50,
        )
        best_threshold = optimized_res.x[0]
        best_coeff = examined_thresholds[best_threshold]
        logger.info(
            f"threshold with best coeff = {best_threshold} (coeff={best_coeff})"
        )
        thresholds_with_best_coeff = [
            thr for thr in examined_thresholds if examined_thresholds[thr] == best_coeff
        ]
        logger.info(
            f"other thresholds with the same coefficient = {','.join([str(np.round(thr,3)) for thr in set(thresholds_with_best_coeff)])}"
        )
        best_threshold = (
            np.min(thresholds_with_best_coeff)
            if for_polyploidy
            else np.max(thresholds_with_best_coeff)
        )
        logger.info(f"final selected {'poly' if for_polyploidy else 'di'}ploidy threshold with the same coeff = {best_threshold}")
        # compute reliability scores
        taxon_to_reliability_scores = pd.DataFrame(
            columns=[
                "node",
                "frac_sim_supporting_polyploidy",
                "frac_sim_supporting_diploidy",
            ]
        )

        def get_sim_support_of_ploidy(node: str, ploidy_state: int) -> float:
            num_supporting_sim = len(
                ploidy_data.loc[
                    (ploidy_data.node == node)
                    & (ploidy_data.is_polyploid == ploidy_state),
                    "simulation",
                ].unique()
            )
            num_sim = len(ploidy_data.simulation.unique())
            return num_supporting_sim / num_sim

        taxon_to_reliability_scores["node"] = ploidy_data.node.unique()
        taxon_to_reliability_scores[
            f"frac_sim_supporting_polyploidy"
        ] = taxon_to_reliability_scores["node"].apply(
            lambda node: get_sim_support_of_ploidy(node, ploidy_state=1)
        )
        taxon_to_reliability_scores[
            f"frac_sim_supporting_diploidy"
        ] = taxon_to_reliability_scores["node"].apply(
            lambda node: get_sim_support_of_ploidy(node, ploidy_state=0)
        )
        taxon_to_reliability_scores.reset_index(inplace=True)
        return best_threshold, best_coeff, taxon_to_reliability_scores

    @staticmethod
    def _get_inferred_is_polyploid(
        taxon: str, tree_with_states: Tree, node_to_is_polyploid: dict[str, bool]
    ) -> bool:
        taxon_node = [
            l
            for l in tree_with_states.traverse()
            if l.name.lower().startswith(taxon.lower())
        ][0]
        taxon_chromosomes_number = int(taxon_node.name.split("-")[-1])
        taxon_parent_node = taxon_node.up
        if taxon_parent_node is None:
            return False
        parent_name = taxon_parent_node.name
        if parent_name in node_to_is_polyploid and node_to_is_polyploid[parent_name]:
            return True
        taxon_parent_chromosomes_number = int(taxon_parent_node.name.split("-")[-1])
        if taxon_chromosomes_number >= taxon_parent_chromosomes_number * 1.5:
            return True
        return False

    @staticmethod
    def _process_mappings(
        stochastic_mappings_dir: str,
        tree_with_states_path: str,
        mappings_num: int,
        missing_mappings_threshold: float = 0.8,
    ) -> List[pd.DataFrame]:
        start_time = timer()
        tree_with_states = Tree(tree_with_states_path, format=1)
        num_failed_mappings, num_tomax_mappings, num_considered_mappings = 0, 0, 0
        mappings = []
        for path in os.listdir(stochastic_mappings_dir):
            if path.startswith("stMapping_mapping") and path.endswith(".csv"):
                mapping = pd.read_csv(f"{stochastic_mappings_dir}{path}")
                missing_data_fraction = mapping.isnull().sum().sum() / (
                    mapping.shape[0] * mapping.shape[1]
                )
                if missing_data_fraction > 0.5:
                    num_failed_mappings += 1
                elif np.any(mapping["TOMAX"] >= 1):
                    num_tomax_mappings += 1
                num_considered_mappings += 1
                mapping["ploidity_events_num"] = np.round(
                    mapping[["DUPLICATION", "DEMI-DUPLICATION", "BASE-NUMBER"]].sum(
                        axis=1, skipna=False
                    )
                )
                mapping["is_diploid"] = mapping["ploidity_events_num"] < 1
                mapping["is_polyploid"] = mapping["ploidity_events_num"] >= 1
                node_to_is_polyploid = mapping.set_index("NODE")[
                    "is_polyploid"
                ].to_dict()
                mapping.loc[
                    mapping.ploidity_events_num.isna(), "inferred_is_polyploid"
                ] = mapping.loc[mapping.ploidity_events_num.isna()][
                    ["is_diploid", "NODE", "ploidity_events_num"]
                ].apply(
                    lambda record: Pipeline._get_inferred_is_polyploid(
                        taxon=record.NODE,
                        tree_with_states=tree_with_states,
                        node_to_is_polyploid=node_to_is_polyploid,
                    ),
                    axis=1,
                )
                mapping["inferred_is_diploid"] = 1 - mapping["inferred_is_polyploid"]
                mappings.append(mapping)
        logger.info(
            f"% mappings with failed leaves trajectories = {np.round(num_failed_mappings/mappings_num, 2)}% ({num_failed_mappings} / {mappings_num})"
        )
        logger.info(
            f"% mappings with trajectories that reached max chromosome number = {np.round(num_tomax_mappings / mappings_num, 2)}% ({num_tomax_mappings} / {mappings_num})"
        )
        if num_considered_mappings < mappings_num * missing_mappings_threshold:
            logger.warning(
                f"less than {missing_mappings_threshold*100}% of the mappings were successful, and so the script will halt"
            )
        end_time = timer()
        logger.info(
            f"completed processing {mappings_num} mappings within {stochastic_mappings_dir} in {timedelta(seconds=end_time-start_time)}"
        )
        return mappings

    @staticmethod
    def _create_simulation_input(
        tree_path: str,
        model_parameters_path: str,
        work_dir: str,
        frequencies_path: str,
        min_observed_chr_count: int,
        max_observed_chr_count: int,
        trials_num: int,
        simulations_num: int,
    ) -> str:
        os.makedirs(work_dir, exist_ok=True)
        sm_input_path = f"{work_dir}sim_params.json"
        chromevol_input_path = f"{work_dir}sim.params"
        input_args = {
            "input_path": chromevol_input_path,
            "tree_path": tree_path,
            "output_dir": work_dir,
            "optimize_points_num": 1,
            "optimize_iter_num": 0,
            "frequencies_path": frequencies_path,
            "min_chromosome_num": 1,
            "max_chromosome_num": 200,
            "max_transitions_num": int(max_observed_chr_count - min_observed_chr_count),
            "max_chr_inferred": int(max_observed_chr_count + 10),
            "simulate": True,
            "run_stochastic_mapping": False,
            "num_of_simulation_trials": trials_num,
            "num_of_simulations": simulations_num,
            "allowed_failed_sim_frac": 1.0 - float(simulations_num / trials_num),
        }
        with open(model_parameters_path, "r") as infile:
            selected_model_res = json.load(fp=infile)
            input_args["parameters"] = selected_model_res["model_parameters"]
            input_args["tree_scaling_factor"] = selected_model_res[
                "tree_scaling_factor"
            ]
        with open(sm_input_path, "w") as outfile:
            json.dump(obj=input_args, fp=outfile)
        return sm_input_path

    @staticmethod
    def _parse_simulation_frequencies(model_parameters_path: str, output_path: str):
        with open(model_parameters_path, "r") as infile:
            model = json.load(fp=infile)
        orig_frequencies_path = model["states_frequencies_path"]
        with open(orig_frequencies_path, "rb") as infile:
            orig_frequencies = pickle.load(infile)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as outfile:
            for i in range(1, 201):
                outfile.write(f"{orig_frequencies.get(i, 0)}\n")

    @staticmethod
    def _get_max_state_transition(evol_path: str):
        file = open(evol_path, "r")
        content = file.read()
        file.close()
        pattern_state_to_state = re.compile(
            "from state:[\s]+([\d]+)[\s]+t[\s]+=[\s]+[\S]+[\s]+to state[\s]+=[\s]+([\d]+)"
        )
        states = pattern_state_to_state.findall(content)
        max_state = 0
        for state_from, state_to in states:
            if int(state_from) > max_state:
                max_state = int(state_from)
            if int(state_to) > max_state:
                max_state = int(state_to)
            else:
                continue
        return max_state

    @staticmethod
    def _remove_unsuccessful_simulations(simulations_dir: str, max_state: int = 200):
        sim_folders = os.listdir(simulations_dir)
        for i in sim_folders:
            sim_dir_i = os.path.join(simulations_dir, str(i))
            if (not os.path.exists(sim_dir_i)) or (not os.path.isdir(sim_dir_i)):
                continue
            evol_path = os.path.join(sim_dir_i, "simulatedEvolutionPaths.txt")
            if not os.path.exists(evol_path):
                continue
            max_state_evol = Pipeline._get_max_state_transition(evol_path)
            if max_state_evol == max_state:
                shutil.rmtree(sim_dir_i)

    @staticmethod
    def _pick_successful_simulations(simulations_dir: str, simulations_num: int):
        dirs = os.listdir(simulations_dir)
        lst_of_dirs = []
        for file in dirs:
            sim_i_dir = os.path.join(simulations_dir, file)
            if not os.path.isdir(sim_i_dir):
                continue
            lst_of_dirs.append(int(file))
        lst_of_dirs.sort()
        for i in range(simulations_num):
            curr_dir = os.path.join(simulations_dir, str(lst_of_dirs[i]))
            dst = os.path.join(simulations_dir, str(i))
            if i != lst_of_dirs[i]:
                os.rename(curr_dir, dst)

    @staticmethod
    def _simulate(
        tree_path: str,
        model_parameters_path: str,
        min_observed_chr_count: int,
        max_observed_chr_count: int,
        simulations_dir: str,
        trials_num: int = 1000,
        simulations_num: int = 10,
    ) -> list[str]:
        if not(os.path.exists(simulations_dir) and len(os.listdir(simulations_dir)) >= simulations_num):

            frequencies_path = f"{simulations_dir}/simulations_states_frequencies.txt"
            Pipeline._parse_simulation_frequencies(
                model_parameters_path=model_parameters_path, output_path=frequencies_path
            )
            sm_input_path = Pipeline._create_simulation_input(
                tree_path=tree_path,
                model_parameters_path=model_parameters_path,
                work_dir=simulations_dir,
                frequencies_path=frequencies_path,
                min_observed_chr_count=min_observed_chr_count,
                max_observed_chr_count=max_observed_chr_count,
                trials_num=trials_num,
                simulations_num=simulations_num,
            )

            if len(os.listdir(simulations_dir)) < simulations_num:
                commands = [
                    os.getenv("CONDA_ACT_CMD"),
                    f"cd {simulations_dir}",
                    f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={sm_input_path}",
                ]
                joined_cmd = ";".join(commands).replace("//", "/")
                res = os.system(joined_cmd)

            Pipeline._remove_unsuccessful_simulations(
                simulations_dir=simulations_dir, max_state=200
            )
            Pipeline._pick_successful_simulations(
                simulations_dir=simulations_dir, simulations_num=simulations_num
            )

        successful_simulations_dirs = []
        for path in os.listdir(simulations_dir):
            try:
                is_sim_path = int(path)
                simulation_dir = f"{simulations_dir}{path}/"
                Pipeline._extract_ploidy_levels(simulation_dir=simulation_dir)
                successful_simulations_dirs.append(simulation_dir)
            except Exception as e:
                continue

        if len(successful_simulations_dirs) < simulations_num-1:
            raise ValueError(
                f"after {trials_num} trials, chromevol succeeded in creating only {len(successful_simulations_dirs)} successful simulations"
            )

        return successful_simulations_dirs

    @staticmethod
    def _extract_ploidy_levels(simulation_dir: str):
        node_to_is_polyploid = dict()
        simulated_evolution_path = f"{simulation_dir}/simulatedEvolutionPaths.txt"
        branch_evolution_regex = re.compile(
            "\*\n\s*(.*?)\nFather is\:\s*(.*?)\n.*?#", re.MULTILINE | re.DOTALL
        )
        transition_regex = re.compile("from state\:\s*(\d*).*?to state = (\d*)")
        with open(simulated_evolution_path, "r") as infile:
            branch_evolution_paths = [
                match for match in branch_evolution_regex.finditer(infile.read())
            ]
        for branch_evolution_path in branch_evolution_paths:
            child, parent = (
                branch_evolution_path.group(1).replace("N-", "N"),
                branch_evolution_path.group(2).replace("N-", "N"),
            )
            if node_to_is_polyploid.get(parent, False):
                node_to_is_polyploid[child] = 1
                continue
            for transition in transition_regex.finditer(branch_evolution_path.group(0)):
                src, dst = int(transition.group(1)), int(transition.group(2))
                if src == 200 or dst == 200:
                    res = os.system(f"rm -rf {simulation_dir}")
                    raise ValueError(
                        f"simulation at {simulation_dir} failed due to reaching maximal state 200 and was thus removed"
                    )
                if (
                    abs(src - dst) > 1
                ):  # any increase larger than is either base number duplication / demi-duplication or full duplication
                    node_to_is_polyploid[child] = 1
                    # update children to be polyploids
                    continue
            if child not in node_to_is_polyploid:
                node_to_is_polyploid[child] = 0
        node_to_is_polyploid_path = f"{simulation_dir}/node_to_is_polyploid.csv"
        node_to_is_polyploid_df = (
            pd.DataFrame.from_dict(node_to_is_polyploid, orient="index")
            .reset_index()
            .rename(columns={"index": "node", 0: "is_polyploid"})
        )
        node_to_is_polyploid_df.to_csv(node_to_is_polyploid_path, index=False)

    def _get_simulations(
        self,
        tree_path: str,
        orig_counts_path: str,
        model_parameters_path: str,
        simulations_num: int = 10,
        trials_num: int = 1000,
    ) -> list[str]:
        simulations_dir = f"{self.work_dir}/simulations/"
        counts = [
            int(str(record.seq)) if str(record.seq) != "X" else np.nan
            for record in list(SeqIO.parse(orig_counts_path, format="fasta"))
        ]
        min_observed_chr_count, max_observed_chr_count = (
            int(np.nanmin(counts)),
            int(np.nanmax(counts)),
        )
        successful_simulations = self._simulate(
            tree_path=tree_path,
            model_parameters_path=model_parameters_path,
            min_observed_chr_count=min_observed_chr_count,
            max_observed_chr_count=max_observed_chr_count,
            simulations_dir=simulations_dir,
            trials_num=trials_num,
            simulations_num=simulations_num,
        )
        return successful_simulations

    def _get_simulation_based_thresholds(
        self,
        counts_path: str,
        tree_path: str,
        model_parameters_path: str,
        mappings_num: int,
        simulations_num: int = 10,
        trials_num: int = 1000,
        debug: bool = False,
    ) -> Tuple[
        float, float, pd.DataFrame, pd.DataFrame,
    ]:
        simulations_dirs = self._get_simulations(
            orig_counts_path=counts_path,
            tree_path=tree_path,
            model_parameters_path=model_parameters_path,
            simulations_num=simulations_num,
            trials_num=trials_num,
        )

        # for each one, do mappings and then extract frequencies of poly events per species
        logger.info(
            f"creating stochastic mappings per simulation to compute duplication events frequencies"
        )
        simulations_ploidy_data_path = f"{self.work_dir}/simulations/simulations_ploidy_data_on_{simulations_num}_simulations.csv"
        sim_num_to_thresholds_path = {
            simulations_num: f"{self.work_dir}/simulations/{simulations_num}_simulations_based_thresholds.pkl"
        }
        if debug:
            for i in range(10, simulations_num, 10):
                sim_num_to_thresholds_path[
                    i
                ] = f"{self.work_dir}/simulations/{i}_simulations_based_thresholds.pkl"

        if not os.path.exists(simulations_ploidy_data_path):
            input_to_output = dict()
            simulations_ploidy_data = []
            sm_input_path = None
            for simulation_dir in simulations_dirs:
                sm_work_dir = f"{simulation_dir}/stochastic_mapping/"
                if not os.path.exists(sm_work_dir):
                    os.makedirs(sm_work_dir, exist_ok=True)
                    sm_input_path, sm_output_dir = self._create_stochastic_mapping_input(
                        counts_path=f"{simulation_dir}/counts.fasta",
                        tree_path=tree_path,
                        model_parameters_path=model_parameters_path,
                        work_dir=sm_work_dir,
                        mappings_num=mappings_num,
                        optimize_model=True,
                    )
                    if (
                            not os.path.exists(sm_output_dir)
                            or len(os.listdir(sm_output_dir)) < mappings_num
                    ):
                        input_to_output[sm_input_path] = f"{sm_output_dir}/chromevol.res"

            self._send_chromevol_commands(command_dir=f"{self.work_dir}/simulations/",
                                              input_to_output=input_to_output,
                                              input_to_run_in_main=sm_input_path,
                                              )

            for simulation_dir in simulations_dirs:
                node_to_ploidy_level = pd.read_csv(
                    f"{simulation_dir}/node_to_is_polyploid.csv"
                )
                node_to_duplication_events_frequency = self._get_frequency_of_duplication_events(
                    sm_work_dir=f"{simulation_dir}/stochastic_mapping/",
                    mappings_num=mappings_num,
                )
                node_to_ploidy_level.set_index("node", inplace=True)
                node_to_ploidy_level["duplication_events_frequency"] = np.nan
                node_to_ploidy_level["duplication_events_frequency"].fillna(
                    value=node_to_duplication_events_frequency.set_index("NODE")[
                        "polyploidy_frequency"
                    ].to_dict(),
                    inplace=True,
                )
                node_to_ploidy_level.reset_index(inplace=True)
                output_path = f"{simulation_dir}/mappings_frequencies.csv"
                node_to_ploidy_level.to_csv(output_path, index=False)

                node_to_ploidy_level["simulation"] = simulation_dir
                simulations_ploidy_data.append(node_to_ploidy_level)

            simulations_ploidy_data = pd.concat(simulations_ploidy_data)
            simulations_ploidy_data.to_csv(simulations_ploidy_data_path, index=False)
        else:
            simulations_ploidy_data = pd.read_csv(simulations_ploidy_data_path)

        # learn optimal thresholds based on frequencies
        diploidity_threshold, polyploidity_threshold = np.nan, np.nan
        diploidy_reliability_scores, polyploidy_reliability_scores = np.nan, np.nan
        for sim_num in sim_num_to_thresholds_path:
            thresholds_path = sim_num_to_thresholds_path[sim_num]
            # if os.path.exists(thresholds_path):
            if False:
                with open(thresholds_path, "rb") as infile:
                    optimal_thresholds_data = pickle.load(file=infile)
                    polyploidity_threshold = optimal_thresholds_data[
                        "polyploidity_threshold"
                    ]
                    polyploidy_reliability_scores = optimal_thresholds_data[
                        "polyploidy_reliability_scores"
                    ]
                    diploidity_threshold = optimal_thresholds_data[
                        "diploidity_threshold"
                    ]
                    diploidy_reliability_scores = optimal_thresholds_data[
                        "diploidy_reliability_scores"
                    ]

            else:
                logger.info(
                    f"finding optimal polyploidy and diploidy thresholds based on {sim_num} simulations"
                )
                (
                    polyploidity_threshold,
                    polyploidy_coeff,
                    polyploidy_reliability_scores,
                ) = self._get_optimal_threshold(
                    ploidy_data=simulations_ploidy_data.loc[
                        simulations_ploidy_data.simulation.isin(
                            simulations_dirs[:sim_num]
                        )
                    ],
                    for_polyploidy=True,
                )

                logger.info(
                    f"optimal polyploidy threshold = {polyploidity_threshold} (with coeff={polyploidy_coeff})"
                )

                (
                    diploidity_threshold,
                    diploidy_coeff,
                    diploidy_reliability_scores,
                ) = self._get_optimal_threshold(
                    ploidy_data=simulations_ploidy_data.loc[
                        simulations_ploidy_data.simulation.isin(
                            simulations_dirs[:sim_num]
                        )
                    ],
                    for_polyploidy=False,
                    upper_bound=np.min([0.5, polyploidity_threshold]),
                )

                logger.info(
                    f"optimal diploidy threshold = {diploidity_threshold} (with coeff={diploidy_coeff})"
                )

                optimal_thresholds = {
                    "diploidity_threshold": diploidity_threshold,
                    "diploidity_coeff": diploidy_coeff,
                    "polyploidity_threshold": polyploidity_threshold,
                    "polyploidity_coeff": polyploidy_coeff,
                    "diploidy_reliability_scores": diploidy_reliability_scores,
                    "polyploidy_reliability_scores": polyploidy_reliability_scores,
                }
                with open(thresholds_path, "wb") as outfile:
                    pickle.dump(obj=optimal_thresholds, file=outfile)

            if sim_num == simulations_num:
                with open(thresholds_path, "rb") as infile:
                    optimal_thresholds = pickle.load(file=infile)
                diploidity_threshold = optimal_thresholds["diploidity_threshold"]
                polyploidity_threshold = optimal_thresholds["polyploidity_threshold"]

        return (
            diploidity_threshold,
            polyploidity_threshold,
            diploidy_reliability_scores,
            polyploidy_reliability_scores,
        )

    def _get_frequency_of_duplication_events(
        self,
        sm_work_dir: str,
        mappings_num: int,

    ) -> pd.DataFrame:
        sm_output_dir = f"{sm_work_dir}/stochastic_mappings/"
        processed_mappings_path = f"{sm_work_dir}/processed_mappings.csv"
        if os.path.exists(processed_mappings_path):
            taxon_to_polyploidy_support = pd.read_csv(processed_mappings_path)
        else:
            mappings = self._process_mappings(
            stochastic_mappings_dir=sm_output_dir,
            mappings_num=mappings_num,
            tree_with_states_path=f"{self.work_dir}/stochastic_mapping/MLAncestralReconstruction.tree",
        )
            all_mappings = pd.concat(mappings)
            taxon_to_polyploidy_support = (
                all_mappings[["NODE", "is_polyploid", "inferred_is_polyploid"]]
                .groupby("NODE")
                .agg(
                    {
                        "is_polyploid": lambda labels: np.sum(labels) / len(labels.dropna())
                        if len(labels.dropna()) > 0
                        else np.nan,
                        "inferred_is_polyploid": lambda labels: np.sum(labels)
                        / len(labels.dropna())
                        if len(labels.dropna()) > 0
                        else np.nan,
                    }
                )
                .reset_index()
                .rename(
                    columns={
                        "is_polyploid": "polyploidy_frequency",
                        "inferred_is_polyploid": "inferred_polyploidy_frequency",
                    }
                )
            )
            taxon_to_polyploidy_support.set_index("NODE", inplace=True)
            taxon_to_polyploidy_support["polyploidy_frequency"].fillna(
                value=taxon_to_polyploidy_support[
                    "inferred_polyploidy_frequency"
                ].to_dict(),
                inplace=True,
            )
            taxon_to_polyploidy_support.reset_index(inplace=True)
            taxon_to_polyploidy_support.to_csv(processed_mappings_path, index=False)

        return taxon_to_polyploidy_support

    def _get_frequency_based_ploidity_classification(
        self,
        taxon_to_polyploidy_support: pd.DataFrame,
        polyploidity_threshold: float = 0.9,
        diploidity_threshold: float = 0.1,
    ) -> dict[str, int]:  # 0 - diploid, 1 - polyploid, np.nan - unable to determine

        tree_with_internal_names_path = (
            f"{self.work_dir}/stochastic_mapping/MLAncestralReconstruction.tree"
        )
        full_tree = Tree(tree_with_internal_names_path, format=1)
        for node in full_tree.traverse():
            if "-" in node.name:
                node.name = "-".join(node.name.split("-")[:-1])

        taxon_to_polyploidy_support["is_polyploid"] = (
            taxon_to_polyploidy_support["polyploidy_frequency"]
            >= polyploidity_threshold
        )
        taxon_to_polyploidy_support["is_diploid"] = (
            taxon_to_polyploidy_support["polyploidy_frequency"] <= diploidity_threshold
        )

        # now add ploidy inference by parent
        in_between_names = taxon_to_polyploidy_support.loc[
            (~taxon_to_polyploidy_support.is_polyploid)
            & (~taxon_to_polyploidy_support.is_diploid),
            "NODE",
        ].tolist()
        taxon_to_polyploidy_support.loc[
            taxon_to_polyploidy_support.NODE.isin(in_between_names), "is_polyploid"
        ] = np.nan
        taxon_to_polyploidy_support.loc[
            taxon_to_polyploidy_support.NODE.isin(in_between_names), "is_diploid"
        ] = np.nan

        node_to_is_polyploid = taxon_to_polyploidy_support.set_index("NODE")[
            "is_polyploid"
        ].to_dict()

        def complement_ploidy_inference_by_parent(taxon: str) -> bool:
            taxon_node = [
                l
                for l in full_tree.traverse()
                if l.name.lower().startswith(taxon.lower())
            ][0]
            taxon_parent_node = taxon_node.up
            if taxon_parent_node is None:
                return np.nan
            parent_name = taxon_parent_node.name
            parent_ploidy_level = node_to_is_polyploid.get(parent_name, np.nan)
            taxon_ploidy_level = (
                parent_ploidy_level if parent_ploidy_level == 1 else np.nan
            )
            return taxon_ploidy_level

        taxon_to_polyploidy_support.loc[
            taxon_to_polyploidy_support.is_polyploid.isna(), "is_polyploid"
        ] = taxon_to_polyploidy_support.loc[
            taxon_to_polyploidy_support.is_polyploid.isna(), "NODE"
        ].apply(
            complement_ploidy_inference_by_parent
        )

        taxon_to_polyploidy_support[
            "ploidy_inference"
        ] = taxon_to_polyploidy_support.apply(
            lambda record: 1
            if record.is_polyploid == 1
            else (0 if record.is_diploid == 1 else np.nan),
            axis=1,
        )
        taxon_to_polyploidy_support.NODE = taxon_to_polyploidy_support.NODE

        return taxon_to_polyploidy_support.set_index("NODE")[
            "ploidy_inference"
        ].to_dict()

    def get_ploidity_classification(
        self,
        counts_path: str,
        tree_path: str,
        model_parameters_path: str,
        mappings_num: int = 1000,
        polyploidity_threshold: float = 0.9,
        diploidity_threshold: float = 0.1,
        optimize_thresholds: bool = False,
        taxonomic_classification_data: Optional[pd.DataFrame] = None,
        debug: bool = False,
    ) -> pd.DataFrame:
        ploidy_classification = pd.DataFrame(
            columns=["Taxon", "Genus", "Family", "Ploidy inference"]
        )
        taxa_records = list(SeqIO.parse(counts_path, format="fasta"))
        tree = Tree(tree_path, format=1)
        taxon_name_to_count = {
            record.description: int(str(record.seq))
            if str(record.seq) != "X"
            else np.nan
            for record in taxa_records
        }
        ploidy_classification["Taxon"] = pd.Series(tree.get_leaf_names())
        ploidy_classification["Chromosome count"] = ploidy_classification[
            "Taxon"
        ].apply(lambda name: taxon_name_to_count.get(name, "x"))
        with open(model_parameters_path, "r") as infile:
            parameters = json.load(fp=infile)["model_parameters"]
        if not (
            "dupl" in parameters
            or "demiPloidyR" in parameters
            or "baseNum" in parameters
        ):
            logger.info(
                f"the best selected model has no duplication parameters so all the tip taxa are necessarily diploids"
            )
            ploidy_classification[
                "Ploidy inference"
            ] = 0  # if no duplication parameters are included in the best model, than all taxa must be diploids
        else:
            taxon_to_polyploidy_support = self._get_stochastic_mappings(
                counts_path=counts_path,
                tree_path=tree_path,
                model_parameters_path=model_parameters_path,
                mappings_num=mappings_num,
            )
            polyploidy_reliability_scores, diploidity_reliability_scores = None, None
            if optimize_thresholds:

                simulations_dir = f"{self.work_dir}/simulations/"
                simulations_zip_path = f"{self.work_dir}/simulations.zip"
                if (
                    not os.path.exists(simulations_dir)
                    or os.path.exists(simulations_dir)
                    and len(os.listdir(simulations_dir)) < 10
                ) and os.path.exists(simulations_zip_path):
                    res = os.system(f"cd {self.work_dir}; unzip -o simulations.zip")

                logger.info(f"searching for optimal thresholds based on simulations")
                (
                    diploidity_threshold,
                    polyploidity_threshold,
                    diploidity_reliability_scores,
                    polyploidy_reliability_scores,
                ) = self._get_simulation_based_thresholds(
                    counts_path=counts_path,
                    tree_path=tree_path,
                    model_parameters_path=model_parameters_path,
                    mappings_num=mappings_num,
                    simulations_num=100 if debug else 10,
                    trials_num=1000,
                    debug=debug,
                )
            logger.info(
                f"classifying taxa to ploidity status based on duplication events frequency across stochastic mappings"
            )
            taxon_to_ploidy_classification = self._get_frequency_based_ploidity_classification(
                taxon_to_polyploidy_support=taxon_to_polyploidy_support,
                polyploidity_threshold=polyploidity_threshold,
                diploidity_threshold=diploidity_threshold,
            )

            ploidy_classification.set_index("Taxon", inplace=True)
            ploidy_classification["Ploidy inference"].fillna(
                value=taxon_to_ploidy_classification, inplace=True
            )
            ploidy_classification["Ploidy transitions frequency"] = np.nan
            ploidy_classification["Ploidy transitions frequency"].fillna(value=taxon_to_polyploidy_support.set_index("NODE")["polyploidy_frequency"].to_dict(), inplace=True)

            if optimize_thresholds:
                poly_to_support = polyploidy_reliability_scores.set_index("node")[
                    "frac_sim_supporting_polyploidy"
                ].to_dict()
                di_to_support = diploidity_reliability_scores.set_index("node")[
                    "frac_sim_supporting_diploidy"
                ].to_dict()
                taxon_to_ploidy_classification_support = {}
                for taxon in taxon_to_ploidy_classification:
                    label = taxon_to_ploidy_classification[taxon]
                    support = np.nan
                    if label == 1:
                        support = poly_to_support.get(taxon, np.nan)
                        if pd.isna(support):
                            logger.info(
                                f"taxon {taxon} has no support value for diploidy"
                            )
                    elif label == 0:
                        support = di_to_support.get(taxon, np.nan)
                        if pd.isna(support):
                            logger.info(
                                f"taxon {taxon} has no support value for diploidy"
                            )
                    taxon_to_ploidy_classification_support[taxon] = support
                ploidy_classification["Ploidy inference support"] = np.nan
                ploidy_classification["Ploidy inference support"].fillna(
                    value=taxon_to_ploidy_classification_support, inplace=True
                )

        if taxonomic_classification_data is not None:
            taxon_to_genus, taxon_to_family = (
                taxonomic_classification_data.set_index("taxon")["genus"].to_dict(),
                taxonomic_classification_data.set_index("taxon")["family"].to_dict(),
            )
            ploidy_classification.reset_index(inplace=True)
            ploidy_classification["Genus"] = ploidy_classification.Taxon.apply(
                lambda name: taxon_to_genus.get(name.lower().replace("_", " "), np.nan)
            )
            ploidy_classification["Family"] = ploidy_classification.Taxon.apply(
                lambda name: taxon_to_family.get(name.lower().replace("_", " "), np.nan)
            )

        if os.path.exists(f"{self.work_dir}simulations/"):
            res = os.system(
                f"cd {os.path.dirname(self.work_dir)};zip -r simulations.zip ./simulations"
            )
            res = os.system(f"rm -rf {self.work_dir}simulations/")

        ploidy_classification["Chromosome count"].fillna("x", inplace=True)
        ploidy_classification["Taxon"].replace({"": np.nan}, inplace=True)
        ploidy_classification.dropna(subset=["Taxon"], inplace=True)

        logger.info(
            f"out of {ploidy_classification.shape[0]} taxa, {ploidy_classification.loc[ploidy_classification['Ploidy inference'] == 1].shape[0]} were classified as polyploids, {ploidy_classification.loc[ploidy_classification['Ploidy inference'] == 0].shape[0]} were classified as diploids and {ploidy_classification.loc[ploidy_classification['Ploidy inference'].isna()].shape[0]} have no reliable classification"
        )

        return ploidy_classification

    @staticmethod
    def parse_classification_data(
        ploidy_classification_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], str]:
        labels_str = '<labels>\n\
                                    <label type="text">\n\
                                      <data tag="chrom"/>\n\
                                    </label>\n\
                                 </labels>'
        (
            taxon_to_chromosome_count,
            taxon_to_ploidy_colortag,
            taxon_to_ploidy_class_name,
        ) = (dict(), dict(), dict())
        if ploidy_classification_data is not None:
            taxon_to_chromosome_count = ploidy_classification_data.set_index("Taxon")[
                "Chromosome count"
            ].to_dict()
            taxon_to_ploidy_class = ploidy_classification_data.set_index("Taxon")[
                "Ploidy inference"
            ].to_dict()
            taxon_to_ploidy_colortag = {
                taxon: "#ff0000"
                if taxon_to_ploidy_class[taxon] == 1
                else ("#0000ff" if taxon_to_ploidy_class[taxon] == 0 else "0x000000")
                for taxon in taxon_to_ploidy_class
            }
            taxon_to_ploidy_class_name = {
                taxon: "Polyploid"
                if taxon_to_ploidy_class[taxon] == 1
                else ("Diploid" if taxon_to_ploidy_class[taxon] == 0 else "")
                for taxon in taxon_to_ploidy_class
            }
            labels_str = '<labels>\n\
                                <label type="text">\n\
                                  <data tag="chrom"/>\n\
                                </label>\n\
                                <label type="text">\n\
                                  <data tag="ploidy"/>\n\
                                </label>\n\
                                <label type="color">\n\
                                  <data tag="colortag"/>\n\
                                </label>\n\
                             </labels>'
        return (
            taxon_to_chromosome_count,
            taxon_to_ploidy_colortag,
            taxon_to_ploidy_class_name,
            labels_str,
        )

    @staticmethod
    def parse_phyloxml_tree(
        input_path: str,
        output_path: str,
        taxon_to_chromosome_count: Dict[str, str],
        taxon_to_ploidy_colortag: Dict[str, str],
        taxon_to_ploidy_class_name: Dict[str, str],
        labels_str: str,
    ):

        with open(input_path, "r") as phylo_in:
            phylo_tree_lines = phylo_in.readlines()

        with open(output_path, "w") as phylo_out:
            phylo_out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            for line in phylo_tree_lines:
                if "<name>" in line:
                    taxon_name_with_chromosome_count = re.split(">|<|\n", line)[2]
                    taxon_name = taxon_name_with_chromosome_count.split("-")[0]
                    phylo_out.write(
                        line.replace(taxon_name_with_chromosome_count, taxon_name)
                    )
                    chrom_tag = "<chrom> - </chrom>\n"
                    if taxon_name in taxon_to_chromosome_count:
                        chrom_tag = f"<chrom>{str(taxon_to_chromosome_count.get(taxon_name, 'x')).replace('x', ' - ')}</chrom>\n"
                    phylo_out.write(chrom_tag)
                    if taxon_name in taxon_to_ploidy_colortag:
                        phylo_out.write(
                            f"<colortag>{taxon_to_ploidy_colortag[taxon_name]}</colortag>\n"
                        )
                    if taxon_name in taxon_to_ploidy_class_name:
                        phylo_out.write(
                            f"<ploidy>{taxon_to_ploidy_class_name[taxon_name]}</ploidy>\n"
                        )
                elif "<branch_length>" in line:
                    phylo_out.write(line)
                    if taxon_name:
                        phylo_out.write(f"<name>{taxon_name}<name>")
                elif "<phylogeny" in line:
                    phylo_out.write(line)
                    phylo_out.write(labels_str)
                else:
                    phylo_out.write(line)
                taxon_name = None

    @staticmethod
    def _write_init_phyloxml_tree(newick_path: str, phyloxml_path: str):
        tree = Tree(newick_path, format=1)
        tree.write(outfile=newick_path)
        Phylo.convert(newick_path, "newick", phyloxml_path, "phyloxml")

        # correct names
        for leaf in tree.get_leaves():
            leaf.name = leaf.name
        tree.write(outfile=newick_path)

    @staticmethod
    def write_labeled_phyloxml_tree(
        tree_path: str,
        output_path: str,
        ploidy_classification_data: Optional[pd.DataFrame] = None,
    ):

        init_phyloxml_path = f"{tree_path.split('.')[0]}.phyloxml"
        Pipeline._write_init_phyloxml_tree(
            newick_path=tree_path, phyloxml_path=init_phyloxml_path
        )
        (
            taxon_to_chromosome_count,
            taxon_to_ploidy_colortag,
            taxon_to_ploidy_class_name,
            labels_str,
        ) = Pipeline.parse_classification_data(
            ploidy_classification_data=ploidy_classification_data
        )
        Pipeline.parse_phyloxml_tree(
            input_path=init_phyloxml_path,
            output_path=output_path,
            taxon_to_chromosome_count=taxon_to_chromosome_count,
            taxon_to_ploidy_colortag=taxon_to_ploidy_colortag,
            taxon_to_ploidy_class_name=taxon_to_ploidy_class_name,
            labels_str=labels_str,
        )

    @staticmethod
    def write_labeled_newick_tree(
        tree_path: str,
        output_path: str,
        ploidy_classification_data: Optional[pd.DataFrame] = None,
    ):
        class_to_color = {
            np.nan: "black",
            1: "red",
            0: "blue",
            "polyploid": "red",
            "diploid": "blue",
        }
        tree = Tree(tree_path, format=1)
        for leaf in tree.get_leaves():
            if ploidy_classification_data is not None:
                try:
                    ploidy_status = (
                        ploidy_classification_data.loc[
                            ploidy_classification_data.Taxon == leaf.name,
                            "Ploidy inference",
                        ]
                        .dropna()
                        .values[0]
                    )
                except Exception as e:
                    logger.info(f"no ploidy status is available for {leaf.name}")
                    ploidy_status = np.nan
            else:
                ploidy_status = np.nan
            leaf.add_feature(
                pr_name="color_tag",
                pr_value=f"[&&NHX:C={class_to_color[ploidy_status]}]",
            )
        tree.write(outfile=output_path, features=["color_tag"])