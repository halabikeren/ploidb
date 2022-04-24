import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_generation.chromevol import  model_parameters, models, ChromevolOutput
from services.pbs_service import PBSService

import logging
logger = logging.getLogger(__name__)



class Pipeline:
    work_dir: str

    def __init__(self, work_dir: str = f"{os.getcwd()}/ploidb_pipeline/"):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    @staticmethod
    def create_models_input_files(tree_path: str, counts_path: str, work_dir: str) -> Dict[str, Dict[str, str]]:
        model_to_io = dict()
        base_input_ags = {"tree_path": tree_path,
                          "counts_path": counts_path,
                          "_optimize": True,
                          "_simulate": False,
                          "_num_of_simulations": 0,
                          "_run_stochastic_mapping": False}
        for model in models:
            model_name = "_".join([param.name for param in model])
            model_dir = f"{os.path.abspath(work_dir)}/{model_name}/"
            os.makedirs(model_dir, exist_ok=True)
            model_input_args = base_input_ags.copy()
            model_input_args["input_path"] = f"{model_dir}input.params"
            model_input_args["output_dir"] = model_dir
            model_input_args["parameters"] = {param.name: param.init_value for param in model}
            model_input_path = f"{model_dir}input.json"
            with open(model_input_path, "w") as infile:
                json.dump(obj=model_input_args, fp=infile)
            model_to_io[model_name] = {"input_path": model_input_path, "output_path": f"{model_dir}/parsed_output.json"}
        logger.info(f"created {len(models)} model input files for all possible models across the parameter set {','.join([param.name for param in model_parameters])}")
        return model_to_io

    @staticmethod
    def select_best_model(model_to_io: Dict[str, Dict[str, str]]) -> str:
        winning_model = np.nan
        winning_model_aicc_score = float("-inf")
        for model_name in model_to_io:
            model_output_path = model_to_io[model_name]["output_path"]
            with open(model_output_path, "r") as outfile:
                model_output = ChromevolOutput(**json.load(fp=outfile))
            model_aicc_score = model_output.aicc_score
            if model_aicc_score > winning_model_aicc_score:
                winning_model, winning_model_aicc_score = model_output_path, model_aicc_score
        return winning_model

    def get_best_model(self, counts_path: str, tree_path: str) -> str:
        model_selection_work_dir = f"{self.work_dir}/model_selection/"
        os.makedirs(model_selection_work_dir, exist_ok=True)
        model_to_io = self.create_models_input_files(tree_path=tree_path, counts_path=counts_path, work_dir=model_selection_work_dir)
        jobs_commands = [[os.getenv("CONDA_ACT_CMD"), f"cd {model_selection_work_dir}", f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={model_to_io[model_name]['input_path']}"] for model_name in model_to_io if not os.path.exists(model_to_io[model_name]['output_path'])]
        if len(jobs_commands) > 0:
            PBSService.execute_job_array(work_dir=f"{model_selection_work_dir}jobs/", jobs_commands=jobs_commands, output_dir=f"{model_selection_work_dir}jobs_output/")
        logger.info(f"completed execution of chromevol across {len(jobs_commands)} different models")
        return self.select_best_model(model_to_io)

    @staticmethod
    def create_simulations_input(counts_path: str, tree_path: str, model_parameters_path: str, work_dir: str, simulations_num: int) -> Tuple[str, str]:
        simulation_input_path = f"{work_dir}/simulation_params.json"
        simulations_output_path = f"{work_dir}/expectations_root_to_tip.csv"
        input_ags = {"input_path": counts_path,
                     "tree_path": tree_path,
                     "counts_path": f"{work_dir}/simulated_counts.fasta",
                     "output_dir": work_dir,
                     "_optimize": True,
                     "_simulate": False, # no need to simulate chromosome numbers, only to compute expectations
                     "run_stochastic_mapping": False,
                     "_num_of_simulations": simulations_num}
        with open(model_parameters_path, "r") as infile:
            parameters = json.load(fp=infile)["model_parameters"]
        input_ags["parameters"] = parameters
        with open(simulation_input_path, "w") as outfile:
            json.dump(obj=input_ags, fp=outfile)
        return simulation_input_path, simulations_output_path

    def get_expected_events_num(self, counts_path: str, tree_path: str, model_parameters_path: str, simulations_num: int = 100) -> pd.DataFrame:
        simulations_work_dir = f"{self.work_dir}/simulations/"
        os.makedirs(simulations_work_dir, exist_ok=True)
        simulations_input_path, simulations_output_path = self.create_simulations_input(counts_path=counts_path, tree_path=tree_path, model_parameters_path=model_parameters_path, work_dir=simulations_work_dir, simulations_num=simulations_num)
        PBSService.execute_job_array(work_dir=simulations_work_dir, output_dir=simulations_work_dir, jobs_commands=[[os.getenv("CONDA_ACT_CMD"), f"cd {simulations_work_dir}", f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={simulations_input_path}"]])
        logger.info(f"computed expectations based on {simulations_num} simulations successfully")
        return pd.read_csv(simulations_output_path)

    @staticmethod
    def create_stochastic_mapping_input(counts_path: str, tree_path: str, model_parameters_path: str, work_dir: str, mappings_num: int) -> Tuple[str, str]:
        sm_input_path = f"{work_dir}/sm_params.json"
        sm_output_dir = f"{work_dir}/stochastic_mappings/"
        input_ags = {"input_path": counts_path,
                     "tree_path": tree_path,
                     "counts_path": f"{work_dir}/simulated_counts.fasta",
                     "output_dir": work_dir,
                     "_optimize": True,
                     "_simulate": False, # no need to simulate chromosome numbers, only to compute expectations
                     "run_stochastic_mapping": True,
                     "num_of_simulations": mappings_num}
        with open(model_parameters_path, "r") as infile:
            parameters = json.load(fp=infile)["model_parameters"]
        input_ags["parameters"] = parameters
        with open(sm_input_path, "w") as outfile:
            json.dump(obj=input_ags, fp=outfile)
        return sm_input_path, sm_output_dir

    def create_stochastic_mappings(self, counts_path: str, tree_path: str, model_parameters_path: str, mappings_num: int = 100) -> str:
        sm_work_dir = f"{self.work_dir}/stochastic_mapping/"
        os.makedirs(sm_work_dir, exist_ok=True)
        sm_input_path, sm_output_dir = self.create_stochastic_mapping_input(counts_path=counts_path,
                                                                                        tree_path=tree_path,
                                                                                        model_parameters_path=model_parameters_path,
                                                                                        work_dir=sm_work_dir,
                                                                                        mappings_num=mappings_num)
        PBSService.execute_job_array(work_dir=sm_work_dir, output_dir=sm_work_dir, jobs_commands=[
            [os.getenv("CONDA_ACT_CMD"), f"cd {sm_work_dir}",
             f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={sm_input_path}"]])
        logger.info(f"computed {mappings_num} mappings successfully")
        return sm_output_dir

    def get_stochastic_mappings_based_thresholds(self, counts_path: str, tree_path: str, model_parameters_path: str, expected_events_num: pd.DataFrame, mappings_num: int = 100) -> Tuple[float, float]:
        stochastic_mappings_dir = self.create_stochastic_mappings(counts_path=counts_path, tree_path=tree_path, model_parameters_path=model_parameters_path, mappings_num=mappings_num)
        for path in os.listdir(stochastic_mappings_dir):
            mapping = pd.read_csv(f"{stochastic_mappings_dir}{path}")



    @staticmethod
    def get_ploidity_classification(counts_path: str, tree_path: str, work_dir: str) -> pd.DataFrame:
        pass



if __name__ == '__main__':

    test_work_dir = "/groups/itay_mayrose/halabikeren/PloiDB/chromevol/test/pipeline/"
    os.makedirs(test_work_dir, exist_ok=True)
    test_counts_path = "/groups/itay_mayrose/halabikeren/PloiDB/chromevol/test/counts.fasta"
    test_tree_path = "/groups/itay_mayrose/halabikeren/PloiDB/chromevol/test/tree.newick"

    pipeline = Pipeline(work_dir=test_work_dir)
    logger.info(f"selecting the best chromevol model")
    best_model_results_path = pipeline.get_best_model(counts_path=test_counts_path, tree_path=test_tree_path)
    sim_based_expected_events_num = pipeline.get_expected_events_num(counts_path=test_counts_path, tree_path=test_tree_path, model_parameters_path=best_model_results_path, simulations_num=2)




