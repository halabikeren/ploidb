import dataclasses
import json
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Tuple
from io import StringIO
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class ModelParameter(Enum):
    gain = "gain", "gainFunc", 2.
    loss = "loss", "lossFunc", 2.
    dupl = "dupl", "duplFunc", 2.
    demi_dupl = "demiPloidyR", "demiDuplFunc", 1.3
    base_num = "baseNumR", "baseNumRFunc", 4.

    def __init__(self, name: str, func_name: str, init_value: float):
        self.name = name
        self.func_name = func_name
        self.init_value = init_value



included_parameter_template = "_{param_name}_1 = {param_index};{param_init_value}\n_{func_name} = CONST"
excluded_parameter_template = "{func_name} = IGNORE"

@dataclass
class ChromevolInput:
    tree_path: str
    counts_path: str
    output_dir: str
    input_path: str
    parameters: Dict[str, float] # map of parameter name to its initial value
    _optimize: bool = True
    _optimize_points_num: str = "10, 3, 1"
    _optimize_iter_num: str = "0, 2, 5"
    _simulate: bool = False
    _run_stochastic_mapping: bool = False
    _num_of_simulations: int = 1000


@dataclass
class ChromevolOutput:
    result_path: str
    expected_events_path: str
    stochastic_mappings_path: str
    root_chromosome_number: int
    log_likelihood: float
    aicc_score: float
    model_parameters: Dict[str, float]


class ChromevolExecutor:

    @staticmethod
    def _get_input(input_args: Dict[str, str]) -> ChromevolInput:
        chromevol_input = ChromevolInput(**input_args)
        with open("chromevol_param.template", "r") as infile:
            input_template = infile.read()
        input_string = input_template.format(**dataclasses.asdict(chromevol_input))
        parameters = input_args["parameters"] # ASK TAL HOW TO DO THIS BETTER - I DON'T LIKE THIS
        param_index = 1
        for param in ModelParameter:
            if param.name in parameters:
                input_string += included_parameter_template.format(param_name=param.name, param_index=param_index, param_init_value=input_args["parameters"][param.name], func_name=param.func_name)
            else:
                input_string += excluded_parameter_template.format(func_name=param.func_name)
        with open(chromevol_input.input_path, "w") as outfile:
            outfile.write(input_string)
        return chromevol_input

    @staticmethod
    def _exec(chromevol_input_path: str) -> int:
        cmd = f"{os.getenv('CHROMEVOL_EXEC')}  param={chromevol_input_path}"
        return os.system(cmd)

    @staticmethod
    def _get_model_parameters(result_str: str) -> Dict[str, float]:
        parameters_regex = re.compile("Final model parameters are\:(.*?)AICc", re.MULTILINE | re.DOTALL)
        try:
            parameters_str = parameters_regex.search(result_str).group(1)
            parameter_regex = re.compile("Chromosome\.(.*?)0_1\s=\s(\d*\.*\d*)")
            parameters = dict()
            for match in parameter_regex.finditer(parameters_str):
                parameters[match.group(1)] = float(match.group(2))
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
        log_likelihood_regex = re.compile("Final optimized likelihood is\:\s(\d*\.?\d*)")
        try:
            return float(log_likelihood_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(f"failed to extract log likelihood from {result_str} due to error {e}")
            return np.nan

    @staticmethod
    def _get_aicc_score(result_str: str) -> float:
        aicc_score_regex = re.compile("AICc of the best model\s=\s(\d*\.?\d*)")
        try:
            return float(aicc_score_regex.search(result_str).group(1))
        except Exception as e:
            logger.error(f"failed to extract AICc score from {result_str} due to error {e}")
            return np.nan

    @staticmethod
    def _parse_result(path: str) -> Tuple[float, float, int, Dict[str, float]]:
        with open(path, "r") as infile:
            result_str = infile.read()
        maximum_likelihood_estimators = ChromevolExecutor._get_model_parameters(result_str=result_str)
        root_chromosome_number = ChromevolExecutor._get_root_chromosome_number(result_str=result_str)
        log_likelihood = ChromevolExecutor._get_log_likelihood(result_str=result_str)
        aicc_score = ChromevolExecutor._get_aicc_score(result_str=result_str)
        return log_likelihood, aicc_score, root_chromosome_number, maximum_likelihood_estimators

    @staticmethod
    def _parse_expectations(input_path: str, output_path: str) -> str:
        with open(input_path, "r") as infile:
            expectations = infile.read()
        expectations_regex = re.compile("#EXPECTED NUMBER OF EVENTS FROM ROOT TO LEAF(.*?)\#", re.MULTILINE | re.DOTALL)
        try:
            expectations_str = expectations_regex.search(expectations).group(1)
            expectations_data = pd.read_csv(StringIO(expectations_str), sep="\t")
            expectations_data.to_csv(output_path, index=False)
        except Exception as e:
            logger.error(f"failed to extract expectations data from {input_path} due to error {e}")

    @staticmethod
    def _parse_stochastic_mappings(path: str) -> pd.DataFrame:
        pass

    @staticmethod
    def _parse_output(output_dir: str) -> ChromevolOutput:
        result_path = f"{output_dir}/chromEvol.res"
        expected_events_path = f"{output_dir}/expectations_root_to_tip.csv"
        stochastic_mappings_path = f"{output_dir}/XXXX"
        log_likelihood, aicc_score, root_chromosome_number, model_parameters = ChromevolExecutor._parse_result(path=result_path)
        ChromevolExecutor._parse_expectations(input_path=f"{output_dir}/expectations_second_round.txt", output_path=expected_events_path)
        stochastic_mappings = ChromevolExecutor._parse_stochastic_mappings(path=stochastic_mappings_path)
        stochastic_mappings.to_csv(f"{output_dir}/stochastic_mappings.csv", index=False)
        chromevol_output = ChromevolOutput(result_path=result_path,
                                           expected_events_path=expected_events_path,
                                           stochastic_mappings_path=stochastic_mappings_path,
                                           root_chromosome_number=root_chromosome_number,
                                           log_likelihood=log_likelihood,
                                           aicc_score=aicc_score,
                                           model_parameters=model_parameters)
        with open(f"{output_dir}/parsed_output.json") as outfile:
            json.dump(obj=dataclasses.asdict(chromevol_output), fp=outfile)
        return chromevol_output

    @staticmethod
    def run(input_args: Dict[str, str]) -> ChromevolOutput:
        chromevol_input = ChromevolExecutor._get_input(input_args=input_args)
        ChromevolExecutor._exec(chromevol_input_path=chromevol_input.input_path)
        chromevol_output = ChromevolExecutor._parse_output(chromevol_input.output_dir)
        return chromevol_output

