import json
import os
import re
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import sys

from Bio import SeqIO, Phylo
from ete3 import Tree

from sklearn.metrics import matthews_corrcoef

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
    def _create_models_input_files(tree_path: str, counts_path: str, work_dir: str) -> Dict[str, Dict[str, str]]:
        model_to_io = dict()
        base_input_ags = {"tree_path": tree_path,
                          "counts_path": counts_path,
                          "num_of_simulations": 0,
                          "run_stochastic_mapping": False}
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
        logger.info(f"created {len(models)} model input files for the considered models across the parameter set {','.join([param.name for param in model_parameters])}")
        return model_to_io

    @staticmethod
    def _select_best_model(model_to_io: Dict[str, Dict[str, str]]) -> str:
        winning_model = np.nan
        winning_model_aicc_score = float("inf")
        for model_name in model_to_io:
            model_output_path = model_to_io[model_name]["output_path"]
            with open(model_output_path, "r") as outfile:
                model_output = ChromevolOutput(**json.load(fp=outfile))
            model_aicc_score = model_output.aicc_score
            if model_aicc_score < winning_model_aicc_score:
                winning_model, winning_model_aicc_score = model_output_path, model_aicc_score
        logger.info(f"the selected model is {winning_model} with an AICc score of {winning_model_aicc_score}")
        return winning_model

    def get_best_model(self, counts_path: str, tree_path: str) -> str:
        model_selection_work_dir = f"{self.work_dir}/model_selection/"
        os.makedirs(model_selection_work_dir, exist_ok=True)
        model_to_io = self._create_models_input_files(tree_path=tree_path, counts_path=counts_path, work_dir=model_selection_work_dir)
        jobs_commands = [[os.getenv("CONDA_ACT_CMD"), f"cd {model_selection_work_dir}", f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={model_to_io[model_name]['input_path']}"] for model_name in model_to_io if not os.path.exists(model_to_io[model_name]['output_path'])]
        if len(jobs_commands) > 0:
            PBSService.execute_job_array(work_dir=f"{model_selection_work_dir}jobs/", jobs_commands=jobs_commands, output_dir=f"{model_selection_work_dir}jobs_output/")
        logger.info(f"completed execution of chromevol across {len(jobs_commands)} different models")
        return self._select_best_model(model_to_io)

    @staticmethod
    def _create_simulations_input(counts_path: str, tree_path: str, model_parameters_path: str, work_dir: str, simulations_num: int) -> Tuple[str, str]:
        simulation_input_path = f"{work_dir}simulation_params.json"
        chromevol_input_path = f"{work_dir}simulations.params"
        simulations_output_path = f"{work_dir}expectations_root_to_tip.csv"
        input_ags = {"input_path": chromevol_input_path,
                     "tree_path": tree_path,
                     "counts_path": counts_path,
                     "output_dir": work_dir,
                     "optimize_points_num": 1,
                     "optimize_iter_num": 0,
                     "run_stochastic_mapping": False,
                     "num_of_simulations": simulations_num}
        with open(model_parameters_path, "r") as infile:
            additional_input = json.load(fp=infile)
        parameters = additional_input["model_parameters"]
        input_ags["parameters"] = parameters
        input_ags["tree_scaling_factor"] = additional_input["tree_scaling_factor"]
        with open(simulation_input_path, "w") as outfile:
            json.dump(obj=input_ags, fp=outfile)
        return simulation_input_path, simulations_output_path

    def _get_expected_events_num(self, counts_path: str, tree_path: str, model_parameters_path: str, simulations_num: int = 100) -> pd.DataFrame:
        simulations_work_dir = f"{self.work_dir}simulations/"
        os.makedirs(simulations_work_dir, exist_ok=True)
        simulations_input_path, simulations_output_path = self._create_simulations_input(counts_path=counts_path, tree_path=tree_path, model_parameters_path=model_parameters_path, work_dir=simulations_work_dir, simulations_num=simulations_num)
        if not os.path.exists(simulations_output_path):
            PBSService.execute_job_array(work_dir=f"{simulations_work_dir}jobs/", output_dir=f"{simulations_work_dir}jobs_output/", jobs_commands=[[os.getenv("CONDA_ACT_CMD"), f"cd {simulations_work_dir}", f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={simulations_input_path}"]])
        logger.info(f"computed expectations based on {simulations_num} simulations successfully")
        expected_events_num = pd.read_csv(simulations_output_path)
        expected_events_num["ploidity_events_num"] = expected_events_num[["DUPLICATION", "DEMI-DUPLICATION", "BASE-NUMBER"]].sum(numeric_only=True, axis=1)
        return expected_events_num

    @staticmethod
    def _create_stochastic_mapping_input(counts_path: str, tree_path: str, model_parameters_path: str, work_dir: str, mappings_num: int) -> Tuple[str, str]:
        sm_input_path = f"{work_dir}sm_params.json"
        chromevol_input_path = f"{work_dir}sm.params"
        sm_output_dir = f"{work_dir}stochastic_mappings/"
        input_ags = {"input_path": chromevol_input_path,
                     "tree_path": tree_path,
                     "counts_path": counts_path,
                     "output_dir": work_dir,
                     "optimize_points_num": 1,
                     "optimize_iter_num": 0,
                     "run_stochastic_mapping": True,
                     "num_of_simulations": mappings_num}
        with open(model_parameters_path, "r") as infile:
            parameters = json.load(fp=infile)["model_parameters"]
        input_ags["parameters"] = parameters
        with open(sm_input_path, "w") as outfile:
            json.dump(obj=input_ags, fp=outfile)
        return sm_input_path, sm_output_dir

    def _get_stochastic_mappings(self, counts_path: str, tree_path: str, model_parameters_path: str, mappings_num: int = 100) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        sm_work_dir = f"{self.work_dir}stochastic_mapping/"
        os.makedirs(sm_work_dir, exist_ok=True)
        sm_input_path, sm_output_dir = self._create_stochastic_mapping_input(counts_path=counts_path,
                                                                             tree_path=tree_path,
                                                                             model_parameters_path=model_parameters_path,
                                                                             work_dir=sm_work_dir,
                                                                             mappings_num=mappings_num)
        if not os.path.exists(sm_output_dir) or len(os.listdir(sm_output_dir)) < mappings_num:
            PBSService.execute_job_array(work_dir=f"{sm_work_dir}jobs/", output_dir=f"{sm_work_dir}jobs_output/", jobs_commands=[
                [os.getenv("CONDA_ACT_CMD"), f"cd {sm_work_dir}",
                 f"python {os.path.dirname(__file__)}/run_chromevol.py --input_path={sm_input_path}"]])
        logger.info(f"computed {mappings_num} mappings successfully")
        mappings = self._process_mappings(stochastic_mappings_dir=sm_output_dir,
                                          mappings_num=mappings_num)
        expected_events_num = pd.read_csv(f"{sm_output_dir}/stMapping_root_to_leaf_exp.csv")
        expected_events_num["ploidity_events_num"] = expected_events_num[
            ["DUPLICATION", "DEMI-DUPLICATION", "BASE-NUMBER"]].sum(numeric_only=True, axis=1)
        return mappings, expected_events_num

    @staticmethod
    def _get_true_values(mappings: List[pd.DataFrame], classification_field: str) -> np.array:
        tip_taxa = mappings[0]["NODE"].to_list()
        true_values = pd.DataFrame(columns=["mapping"] + tip_taxa)
        true_values["mapping"] = list(range(len(mappings)))
        for taxon in tip_taxa:
            true_values[taxon] = true_values["mapping"].apply(
                lambda i: 1 if mappings[i].loc[mappings[i]["NODE"] == taxon, classification_field].values[0] else -1)
        return true_values.set_index("mapping").to_numpy().flatten()

    @staticmethod
    def _get_predicted_values(mappings: List[pd.DataFrame], classification_field: str, events_num_thresholds: float) -> np.array:
        is_upper_bound = True if classification_field == "is_diploid" else False
        tip_taxa = mappings[0]["NODE"].to_list()
        predicted_values = pd.DataFrame(columns=["mapping"] + tip_taxa)
        predicted_values["mapping"] = list(range(len(mappings)))
        for taxon in tip_taxa:
            predicted_values[taxon] = predicted_values["mapping"].apply(lambda i: 1 if (is_upper_bound and mappings[i].loc[mappings[i]["NODE"] == taxon, "ploidity_events_num"].values[0] <= events_num_thresholds) or (not is_upper_bound and mappings[i].loc[mappings[i]["NODE"] == taxon, "ploidity_events_num"].values[0] >= events_num_thresholds) else -1)
        return predicted_values.set_index("mapping").to_numpy().flatten()

    @staticmethod
    def _get_optimal_threshold(mappings: List[pd.DataFrame], classification_field: str, expected_events_num: pd.DataFrame):
        true_values = Pipeline._get_true_values(mappings=mappings, classification_field=classification_field)
        max_ploidity_events_num = expected_events_num["ploidity_events_num"].max()
        best_threshold, best_coeff = np.nan, -1.1
        thresholds = [0.01+0.1*i for i in range(11)]
        for threshold in thresholds:
            events_num_threshold = max_ploidity_events_num * threshold
            predicted_values = Pipeline._get_predicted_values(mappings=mappings, classification_field=classification_field, events_num_thresholds=events_num_threshold)
            coeff = matthews_corrcoef(y_true=true_values, y_pred=predicted_values)
            logger.info(f"matthews correlation coefficient for {classification_field} and threshold of {events_num_threshold}")
            if coeff > best_coeff:
                best_threshold, best_coeff = events_num_threshold, coeff
        return  best_coeff

    @staticmethod
    def _process_mappings(stochastic_mappings_dir: str, mappings_num: int, missing_mappings_threshold: float = 0.8) -> List[pd.DataFrame]:
        num_failed_mappings, num_tomax_mappings, num_considered_mappings = 0, 0, 0
        mappings = []
        for path in os.listdir(stochastic_mappings_dir):
            if path.startswith("stMapping_mapping") and path.endswith(".csv"):
                mapping = pd.read_csv(f"{stochastic_mappings_dir}{path}")
                missing_data_fraction = mapping.isnull().sum().sum() / (mapping.shape[0]*mapping.shape[1])
                if missing_data_fraction > 0.5:
                    num_failed_mappings += 1
                    continue
                elif np.any(mapping["TOMAX"] >= 1):
                    num_tomax_mappings += 1
                    continue
                num_considered_mappings += 1
                mapping["ploidity_events_num"] = mapping["DUPLICATION"] + mapping["DEMI-DUPLICATION"] + mapping["BASE-NUMBER"]
                mapping["is_diploid"] = mapping["ploidity_events_num"] < 1
                mapping["is_polyploid"] = mapping["ploidity_events_num"] >= 1
                mappings.append(mapping)
        logger.info(f"% mappings with failed leaves trajectories = {np.round(num_failed_mappings/mappings_num)}% ({num_failed_mappings} / {mappings_num})")
        logger.info(
            f"% mappings with trajectories that reached max chromosome number = {np.round(num_tomax_mappings / mappings_num)}% ({num_tomax_mappings} / {mappings_num})")
        if num_considered_mappings < mappings_num * missing_mappings_threshold:
            logger.error(f"less than {missing_mappings_threshold*100}% of the mappings were successful, and so the script will halt")
            exit(1)
        return mappings

    def _get_stochastic_mappings_based_thresholds(self, mappings: List[pd.DataFrame], expected_events_num: pd.DataFrame) -> Tuple[float, float]:
        diploidity_threshold = self._get_optimal_threshold(mappings=mappings, classification_field="is_diploid", expected_events_num=expected_events_num)
        polyploidity_threshold = self._get_optimal_threshold(mappings=mappings, classification_field="is_polyploid", expected_events_num=expected_events_num)
        logger.info(
            f"optimal diploidity threshold = {diploidity_threshold}, optimal polyploidity threshold = {polyploidity_threshold}")
        return diploidity_threshold, polyploidity_threshold

    @staticmethod
    def _get_frequency_based_ploidity_classification(taxon: str, mappings: List[pd.DataFrame], polyploidity_threshold: float = 0.9, diploidity_threshold: float = 0.1) -> int: # 0 - diploid, 1 - polyploid, np.nan - unable to determine
        if taxon not in mappings[0].NODE.tolist():
            return np.nan
        num_mappings,  num_polyploid_supporting_mappings = len(mappings), 0
        for mapping in mappings:
            if mapping.loc[mapping.NODE == taxon, "is_polyploid"].values[0]:
                num_polyploid_supporting_mappings += 1
        polyploidity_frequency_across_mappings = num_polyploid_supporting_mappings/num_mappings
        if polyploidity_frequency_across_mappings >= polyploidity_threshold:
            return 1
        elif polyploidity_frequency_across_mappings <= diploidity_threshold:
            return 0
        return np.nan

    def get_ploidity_classification(self,
                                    counts_path: str,
                                    tree_path: str,
                                    full_tree_path: str,
                                    model_parameters_path: str,
                                    mappings_num: int = 1000,
                                    classification_based_on_expectations: bool = False,
                                    polyploidity_threshold: float = 0.9,
                                    diploidity_threshold: float = 0.1,
                                    taxonomic_classification_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        ploidy_classification = pd.DataFrame(columns=["Taxon", "Genus", "Family", "Ploidy inference"])
        taxa_records = list(SeqIO.parse(counts_path, format="fasta"))
        tree = Tree(full_tree_path)
        taxon_name_to_count = {record.id: int(str(record.seq)) for record in taxa_records}
        ploidy_classification["Taxon"] = pd.Series(tree.get_leaf_names()).str.lower()
        ploidy_classification["Chromosome count"] = ploidy_classification["Taxon"].apply(lambda name: taxon_name_to_count.get(name, "x"))
        with open(model_parameters_path, "r") as infile:
            parameters = json.load(fp=infile)["model_parameters"]
        if not ("dupl" in parameters or "demiPloidyR" in parameters or "baseNum" in parameters):
            logger.info(f"the best selected model has no duplication parameters so all the tip taxa are necessarily diploids")
            ploidy_classification["Ploidy inference"] = "diploid" # if no duplication parameters are included in the best model, than all taxa must be diploids
        else:
            mappings, expected_events_num = self._get_stochastic_mappings(counts_path=counts_path, tree_path=tree_path,
                                                                    model_parameters_path=model_parameters_path,
                                                                    mappings_num=mappings_num)
            if classification_based_on_expectations:
                logger.info(
                    f"classifying taxa to ploidity status based on thresholds derived from expectations of duplication events")
                diploidity_threshold, polyploidity_threshold = self._get_stochastic_mappings_based_thresholds(mappings=mappings, expected_events_num=expected_events_num)
                ploidy_classification["Ploidy inference"] = ploidy_classification["Taxon"].apply(lambda taxon: np.nan if taxon not in expected_events_num.NODE.tolist() else (0 if expected_events_num.loc[expected_events_num.NODE == taxon, "ploidity_events_num"].values[0] <= diploidity_threshold else (1 if expected_events_num.loc[expected_events_num.NODE == taxon, "ploidity_events_num"].values[0] >= polyploidity_threshold else np.nan)))
            else:
                logger.info(f"classifying taxa to ploidity status based on duplication events frequency across stochastic mappings")
                ploidy_classification["Ploidy inference"] = ploidy_classification["Taxon"].apply(lambda taxon: Pipeline._get_frequency_based_ploidity_classification(taxon=taxon, mappings=mappings, polyploidity_threshold=polyploidity_threshold, diploidity_threshold=diploidity_threshold))
        logger.info(
            f"out of {ploidy_classification.shape[0]} taxa, {ploidy_classification.loc[ploidy_classification['Ploidy inference'] == 1].shape[0]} were classified as polyploids, {ploidy_classification.loc[ploidy_classification['Ploidy inference'] == 0].shape[0]} were classified as diploids and {ploidy_classification.loc[ploidy_classification['Ploidy inference'].isna()].shape[0]} have no reliable classification")
        if taxonomic_classification_data is not None:
            ploidy_classification.set_index("Taxon", inplace=True)
            taxon_to_genus, taxon_to_family = taxonomic_classification_data.set_index("original_name")["genus"].to_dict(), taxonomic_classification_data.set_index("query")["family"].to_dict()
            ploidy_classification["Genus"].fillna(value=taxon_to_genus, inplace=True)
            ploidy_classification["Family"].fillna(value=taxon_to_family, inplace=True)
            ploidy_classification.reset_index(inplace=True)
        return ploidy_classification

    @staticmethod
    def prune_tree_with_counts(counts_path: str, input_tree_path: str, output_tree_path: str):
        counts = list(SeqIO.parse(counts_path, format="fasta"))
        records_with_counts = [record.id for record in counts]
        tree = Tree(input_tree_path)
        tree.prune(records_with_counts)
        tree.write(outfile=output_tree_path)

    @staticmethod
    def parse_classification_data(ploidy_classification_data: Optional[pd.DataFrame] = None) -> Tuple[
        Dict[str, str], Dict[str, str], Dict[str, str], str]:
        labels_str = '<labels>\n\
                                    <label type="text">\n\
                                      <data tag="chrom"/>\n\
                                    </label>\n\
                                 </labels>'
        taxon_to_chromosome_count, taxon_to_ploidy_colortag, taxon_to_ploidy_class_name = dict(), dict(), dict()
        if ploidy_classification_data is not None:
            taxon_to_chromosome_count = ploidy_classification_data.set_index("Taxon")["Chromosome count"].to_dict()
            taxon_to_ploidy_class = ploidy_classification_data.set_index("Taxon")["Ploidy inference"].to_dict()
            taxon_to_ploidy_colortag = {taxon: "#ff0000" if taxon_to_ploidy_class[taxon] == 1 else ("#0000ff" if taxon_to_ploidy_class[taxon] == 0 else "0x000000") for taxon in taxon_to_ploidy_class}
            taxon_to_ploidy_class_name = {taxon: "Polyploid" if taxon_to_ploidy_class[taxon] == 1 else ("Diploid" if taxon_to_ploidy_class[taxon] == 0 else "") for taxon in taxon_to_ploidy_class}
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
        return taxon_to_chromosome_count, taxon_to_ploidy_colortag, taxon_to_ploidy_class_name, labels_str

    @staticmethod
    def parse_phyloxml_tree(input_path: str,
                            output_path: str,
                            taxon_to_chromosome_count: Dict[str, str],
                            taxon_to_ploidy_colortag: Dict[str, str],
                            taxon_to_ploidy_class_name: Dict[str, str],
                            labels_str: str):

        with open(input_path, "r") as phylo_in:
            phylo_tree_lines = phylo_in.readlines()

        with open(output_path, "w") as phylo_out:
            phylo_out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            for line in phylo_tree_lines:
                if "<name>" in line:
                    taxon_name_with_chromosome_count = re.split('>|<|\n', line)[2]
                    taxon_name = taxon_name_with_chromosome_count.split('-')[0]
                    phylo_out.write(line.replace(taxon_name_with_chromosome_count, taxon_name))
                    chrom_tag = "<chrom> - </chrom>\n"
                    if taxon_name in taxon_to_chromosome_count.keys():
                        chrom_tag = f"<chrom>{taxon_to_chromosome_count[taxon_name].replace('x', ' - ')}</chrom>\n"
                    phylo_out.write(chrom_tag)
                    if taxon_name in taxon_to_ploidy_colortag:
                        phylo_out.write(f"<colortag>{taxon_to_ploidy_colortag[taxon_name]}</colortag>\n")
                    if taxon_name in taxon_to_ploidy_class_name:
                        phylo_out.write(f"<ploidy>{taxon_to_ploidy_class_name[taxon_name]}</ploidy>\n")
                elif "<branch_length>" in line:
                    phylo_out.write(line)
                    if taxon_name:
                        phylo_out.write(f"<name>{taxon_name}<name>")
                        taxon_name = None
                elif "<phylogeny" in line:
                    phylo_out.write(line)
                    phylo_out.write(labels_str)
                else:
                    phylo_out.write(line)
                taxon_name = None
    @staticmethod
    def write_labeled_phyloxml_tree(tree_path: str, output_path: str, ploidy_classification_data: Optional[pd.DataFrame] = None):

        input_path = tree_path.replace(".nwk", "phyloxml")
        Phylo.convert(tree_path, 'newick', input_path, 'phyloxml')
        taxon_to_chromosome_count, taxon_to_ploidy_colortag, taxon_to_ploidy_class_name, labels_str = Pipeline.parse_classification_data(ploidy_classification_data=ploidy_classification_data)
        Pipeline.parse_phyloxml_tree(input_path=input_path,
                            output_path=output_path,
                            taxon_to_chromosome_count=taxon_to_chromosome_count,
                            taxon_to_ploidy_colortag=taxon_to_ploidy_colortag,
                            taxon_to_ploidy_class_name=taxon_to_ploidy_class_name,
                            labels_str=labels_str)

    @staticmethod
    def write_labeled_newick_tree(tree_path: str, output_path: str, ploidy_classification_data: Optional[pd.DataFrame] = None):
        class_to_color = {np.nan: "black", 1: "red", 0: "blue"}
        tree = Tree(tree_path)
        for leaf in tree.get_leaves():
            if ploidy_classification_data is not None:
                try:
                    ploidy_status = ploidy_classification_data.loc[ploidy_classification_data.Taxon == leaf.name, "Ploidy inference"].dropna().values[0]
                except Exception as e:
                    logger.info(f"no ploidy status is available for {leaf.name}")
                    ploidy_status = np.nan
            else:
                ploidy_status = np.nan
            leaf.add_feature(pr_name="color_tag", pr_value=f"[&&NHX:C={class_to_color[ploidy_status]}]")
        tree.write(outfile=output_path, features=["color_tag"])

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), ],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    # reproduce ploidb
    test_work_dir = "/groups/itay_mayrose/halabikeren/PloiDB/chromevol/test/reproduce_ploidb/josef/"
    test_counts_path = "/groups/itay_mayrose/halabikeren/PloiDB/chromevol/test/reproduce_ploidb/josef/counts.fasta"
    test_tree_path = "/groups/itay_mayrose/halabikeren/PloiDB/chromevol/test/reproduce_ploidb/josef/tree.nwk"
    taxonomic_classification_path = "/groups/itay_mayrose/halabikeren/PloiDB/name_resolution/processed_resolved_names.csv"

    os.makedirs(test_work_dir, exist_ok=True)
    pipeline = Pipeline(work_dir=test_work_dir)
    relevant_tree_path = test_tree_path.replace(".nwk", "_only_with_counts.nwk")
    pipeline.prune_tree_with_counts(counts_path=test_counts_path, input_tree_path=test_tree_path, output_tree_path=relevant_tree_path)

    logger.info(f"selecting the best chromevol model")
    best_model_results_path = pipeline.get_best_model(counts_path=test_counts_path, tree_path=relevant_tree_path)

    logger.info(f"searching for optimal classification thresholds")
    taxonomic_classification = pd.read_csv(taxonomic_classification_path)
    test_ploidity_classification = pipeline.get_ploidity_classification(counts_path=test_counts_path,
                                                                        tree_path=relevant_tree_path,
                                                                        full_tree_path=test_tree_path,
                                                                        model_parameters_path=best_model_results_path,
                                                                        mappings_num=1000, classification_based_on_expectations = False, taxonomic_classification_data = taxonomic_classification)
    test_ploidity_classification.to_csv(f"{test_work_dir}ploidy.csv")
    pipeline.write_labeled_phyloxml_tree(tree_path=test_tree_path,
                                         ploidy_classification_data=test_ploidity_classification,
                                         output_path=f"{test_work_dir}/classified_tree.phyloxml")

    pipeline.write_labeled_newick_tree(tree_path=test_tree_path,
                                         ploidy_classification_data=test_ploidity_classification,
                                         output_path=f"{test_work_dir}/classified_tree.newick")



