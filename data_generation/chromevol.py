import dataclasses
import os
from dataclasses import dataclass
from typing import Optional, Dict, Union, Tuple

import numpy as np
import pandas as pd

param_name_to_func_name = {"_gain": "_gainFunc",
                           "_loss": "_lossFunc",
                           "_dupl": "_duplFunc",
                           "_demiPloidyR": "_demiDuplFunc",
                           "_baseNumR": "_baseNumRFunc"}
included_parameter_template = "{param_name}_1 = {param_index};{param_init_value}\n{func_name} = CONST"
excluded_parameter_template = "{func_name} = IGNORE"



@dataclass
class ChromevolInput:
    tree_path: Optional[str]
    counts_path: str
    output_dir: str
    input_path: str
    _gain: Optional[float] = 2.
    _loss: Optional[float] = 2.
    _dupl: Optional[float] = 2.
    _demiPloidyR: Optional[float] = 1.3
    _baseNumR: Optional[float] = 4.
    _optimize: bool = True
    _optimize_points_num: str = "10, 3, 1"
    _optimize_iter_num: str = "0, 2, 5"
    _simulate: bool = False
    _run_stochastic_mapping: bool = False


@dataclass
class ChromevolOutput:
    result_path: str
    expected_events_path: str
    stochastic_mappings_path: str
    aicc_score: float
    model_parameters: Dict[str, float]
    expected_events: pd.DataFrame
    stochastic_mappings: pd.DataFrame



class ChromevolExecutor:

    @staticmethod
    def _get_input(input_args: Dict[str, str]) -> ChromevolInput:
        chromevol_input = ChromevolInput(**input_args)
        with open("chromevol_param.template", "r") as infile:
            input_template = infile.read()
        input_string = input_template.format(**dataclasses.asdict(chromevol_input))
        with open(chromevol_input.input_path, "w") as outfile:
            outfile.write(input_string)
        return chromevol_input

    @staticmethod
    def _exec(chromevol_input_path: str) -> int:
        cmd = f"{os.getenv('CHROMEVOL_EXEC')}  param={chromevol_input_path}"
        return os.system(cmd)

    @staticmethod
    def _parse_result(path: str) -> Tuple[float, Dict[str, float]]:
        aicc_score, maximum_likelihood_estimators = np.nan, dict()
        # parse
        return aicc_score, maximum_likelihood_estimators

    @staticmethod
    def _parse_expectations(path: str) -> pd.DataFrame:
        pass

    @staticmethod
    def _parse_stochastic_mappings(path: str) -> pd.DataFrame:
        pass

    @staticmethod
    def _parse_output(output_dir: str) -> ChromevolOutput:
        result_path = f"{output_dir}/chromEvol.res"
        expected_events_path = f"{output_dir}/expectations_second_round.txt"
        stochastic_mappings_path = f"{output_dir}/XXXX"
        aicc_score, model_parameters = ChromevolExecutor._parse_result(path=result_path)
        expected_events = ChromevolExecutor._parse_expectations(path=expected_events_path)
        stochastic_mappings = ChromevolExecutor._parse_stochastic_mappings(path=stochastic_mappings_path)
        chromevol_output = ChromevolOutput(result_path=result_path,
                                           expected_events_path=expected_events_path,
                                           stochastic_mappings_path=stochastic_mappings_path,
                                           aicc_score=aicc_score,
                                           model_parameters=model_parameters,
                                           expected_events=expected_events,
                                           stochastic_mappings=stochastic_mappings)
        return chromevol_output

    @staticmethod
    def run(input_args: Dict[str, str]) -> ChromevolOutput:
        chromevol_input = ChromevolExecutor._get_input(input_args=input_args)
        ChromevolExecutor._exec(chromevol_input_path=chromevol_input.input_path)
        chromevol_output = ChromevolExecutor._parse_output(chromevol_input.output_dir)
        return chromevol_output


