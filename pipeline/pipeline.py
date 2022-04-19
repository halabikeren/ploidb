import itertools
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from data_generation.chromevol import ModelParameter, ChromevolOutput
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
        model_parameters = [param for param in ModelParameter]
        models = [subset for num_parameters in range(1, len(model_parameters) + 1) for subset in
                  itertools.combinations(model_parameters, num_parameters)]
        base_input_ags = {"tree_path": tree_path,
                          "counts_path": counts_path,
                          "_optimize": True,
                          "_simulate": False,
                          "_run_stochastic_mapping": False}
        for model in models:
            model_name = "_".join([param.name for param in model])
            model_dir = f"{work_dir}/{model_name}"
            model_input_args = base_input_ags.copy()
            model_input_args["input_path"] = f"{model_dir}/input.params"
            model_input_args["output_dir"] = model_dir
            model_input_args["parameters"] = {param.name: param.init_value for param in model}
            model_input_path = f"{model_dir}/input.json"
            with open(model_input_path, "w") as infile:
                json.dump(obj=model_input_args, fp=infile)
            model_to_io[model_name] = {"input_path": model_input_path, "output_path": f"{model_dir}/parsed_output.json"}
        logger.info(f"created {len(models)} model input files for all possible models across the parameter set {','.join([param.name for param in ModelParameter])}")
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
                winning_model_name, winning_model_aicc_score = model_output_path, model_aicc_score
        return winning_model

    def get_best_model(self, counts_path: str, tree_path: str):
        model_selection_work_dir = f"{self.work_dir}/model_selection/"
        os.makedirs(model_selection_work_dir, exist_ok=True)
        model_to_io = self.create_models_input_files(tree_path=tree_path, counts_path=counts_path, work_dir=model_selection_work_dir)
        jobs_commands = [[f"cd {model_selection_work_dir}", f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={model_to_io[model_name]['input_path']}"] for model_name in model_to_io]
        PBSService.execute_job_array(work_dir=f"{model_selection_work_dir}jobs/", jobs_commands=jobs_commands, output_dir=f"{model_selection_work_dir}jobs_output/")
        logger.info(f"completed execution of chromevol across {len(jobs_commands)} different models")
        return self.select_best_model(model_to_io)

    @staticmethod
    def create_simulations_input(tree_path: str, model_parameters_path: str, work_dir: str, simulations_num: int) -> Tuple[str, str]:
        simulation_input_path = f"{work_dir}/simulate.params"
        simulations_output_path = f"{work_dir}/expectations_root_to_tip.csv"
        input_ags = {"tree_path": tree_path,
                     "counts_path": f"{work_dir}/simulated_counts.fasta",
                     "output_dir": work_dir,
                     "_optimize": True,
                     "_simulate": True,
                     "_run_stochastic_mapping": False,
                     "_num_of_simulations": simulations_num}
        with open(model_parameters_path, "r") as infile:
            model_parameters = json.load(fp=infile)["model_parameters"]
        input_ags["parameters"] = model_parameters
        with open(simulation_input_path, "w") as outfile:
            json.dump(obj=input_ags, fp=outfile)
        return simulation_input_path, simulations_output_path

    def get_expected_events_num(self, tree_path: str, model_parameters_path: str, simulations_num: int = 100) -> pd.DataFrame:
        simulations_work_dir = f"{self.work_dir}/simulations/"
        os.makedirs(simulations_work_dir, exist_ok=True)
        simulations_input_path, simulations_output_path = self.create_simulations_input(tree_path=tree_path, model_parameters_path=model_parameters_path, work_dir=simulations_work_dir, simulations_num=simulations_num)
        PBSService.execute_job_array(work_dir=simulations_input_path, output_dir=simulations_work_dir, jobs_commands=[[f"cd {simulations_work_dir}", f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={simulations_input_path}"]])
        logger.info(f"computed expectations based on {simulations_num} simulations successfully")
        return pd.read_csv(simulations_output_path)

    @staticmethod
    def get_stochastic_mapping_based_thresholds(counts_path: str, tree_path: str, output_dir: str, model_parameters: Dict[str, float], mappings_num: int = 100) -> Tuple[float,float]:
        pass

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
    pipeline.get_best_model(counts_path=test_counts_path, tree_path=test_tree_path)




