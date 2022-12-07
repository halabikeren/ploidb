import dataclasses
import os
import re
import subprocess
from dataclasses import dataclass
from time import sleep
from typing import Dict

from dotenv import load_dotenv, find_dotenv
from ete3 import Tree

load_dotenv(find_dotenv())

import logging

logger = logging.getLogger(__name__)

taxa_list_filename = "userInput.txt"
parameters_filename = "params.txt"
email_filename = "email.txt"
empty_file_names = ["ConstraintTree_user.txt", "NodeDate.txt", "email.txt", "ConstraintTaxIdList_user.txt"]


@dataclass
class OneTwoTreeInput:
    taxa_list_path: str
    job_name: str
    output_dir: str
    parameters_path: str
    queue: str

    def __init__(self, taxa_path: str, job_name: str, output_dir: str, queue: str = "itaym"):
        os.makedirs(output_dir, exist_ok=True)
        taxa_list_path = f"{output_dir}{taxa_list_filename}"
        res = os.system(f"cp {taxa_path} {taxa_list_path}")
        self.taxa_list_path = taxa_list_path

        email_path = f"{output_dir}{email_filename}"
        with open(email_path, "w") as f:
            f.write(os.getenv("ONETWOTREE_EMAIL"))

        parameters_path = f"{output_dir}{parameters_filename}"
        self.parameters_path = parameters_path

        self.job_name = job_name
        self.output_dir = output_dir
        self.queue = queue

        for filename in empty_file_names:
            with open(f"{output_dir}{filename}", mode="a"):
                pass


@dataclass
class OneTwoTreeOutput:
    tree_path: str
    covered_taxa_path: str


class OneTwoTreeExecutor:
    @staticmethod
    def _get_input(input_args: Dict[str, str]) -> OneTwoTreeInput:
        one_two_tree_input = OneTwoTreeInput(**input_args)
        with open(f"{os.path.dirname(__file__)}/one_two_tree_param.template", "r") as infile:
            input_template = infile.read()
        input_string = input_template.format(**dataclasses.asdict(one_two_tree_input))
        with open(one_two_tree_input.parameters_path, "w") as outfile:
            outfile.write(input_string)
        return one_two_tree_input

    @staticmethod
    def _exec(
        exe_input: OneTwoTreeInput,
    ) -> int:
        cmd = f"module load python/anaconda3-5.0.0;python {os.path.dirname(os.path.realpath(__file__))}/one_two_tree_bioseq_script.py {exe_input.taxa_list_path} {exe_input.output_dir} {exe_input.job_name} {exe_input.queue};"
        cmd.replace("//", "/")
        res = subprocess.getoutput(cmd)
        job_id = re.search("(\d+)\.power\d", str(res)).group(1)
        while not os.system(f"qstat -f {job_id} > /dev/null 2>&1") != 0:
            sleep(60)
        return 0

    @staticmethod
    def _parse_output(one_two_tree_input: OneTwoTreeInput) -> OneTwoTreeOutput:
        output_path = f"OneTwoTree_Output_{one_two_tree_input.job_name}.zip"
        res = os.system(f"cd {one_two_tree_input.output_dir}; unzip -o {output_path}")
        tree_path = f"{one_two_tree_input.output_dir}Result_Tree_{one_two_tree_input.job_name}.tre"
        covered_taxa_path = f"{one_two_tree_input.output_dir}FinalSpeciesList.txt"
        summary_path = f"{one_two_tree_input.output_dir}summary_file.txt"
        selected_out_group_regex = re.compile("Selected Outgroup\:\s*(.*?)\n", re.DOTALL)
        with open(summary_path, "r") as f:
            out_group_name = selected_out_group_regex.search(f.read()).group(1)
        with open(covered_taxa_path, "r") as f:
            covered_taxa = [s.replace("\n", "") for s in f.readlines()]
        tree = Tree(tree_path, format=5)
        non_out_group_leaves = list(set(tree.get_leaf_names()) - {out_group_name})
        tree.prune(non_out_group_leaves, preserve_branch_length=True)
        for l in tree.get_leaves():
            l.name = l.name.replace("_", " ")
        assert set(tree.get_leaf_names()) == set(covered_taxa)
        output_tree_path = f"{one_two_tree_input.output_dir}/processed_tree.nwk"
        tree.write(outfile=output_tree_path)
        OneTwoTreeOutput(tree_path=output_tree_path, covered_taxa_path=covered_taxa_path)

    @staticmethod
    def run(input_args: Dict[str, str]) -> OneTwoTreeOutput:
        one_two_tree_input = OneTwoTreeExecutor._get_input(input_args=input_args)
        raw_output_path = f"{one_two_tree_input.output_dir}OneTwoTree_Output_{one_two_tree_input.job_name}.zip"
        if not os.path.exists(raw_output_path):
            res = OneTwoTreeExecutor._exec(exe_input=one_two_tree_input)
            if res != 0:
                raise ValueError(
                    f"one two tree failed on {one_two_tree_input.taxa_list_path} with parameters path {one_two_tree_input.parameters_path}"
                )
        if os.path.exists(raw_output_path):
            one_two_tree_output = OneTwoTreeExecutor._parse_output(one_two_tree_input)
            return one_two_tree_output
