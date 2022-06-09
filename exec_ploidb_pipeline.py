import time
from typing import Optional

import click
import logging
from pipeline import Pipeline
import sys
import os
import pandas as pd

from timeit import default_timer as timer
from datetime import timedelta

logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--counts_path",
    help="chromosome counts file path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--tree_path",
    help="path to the tree file",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--output_dir",
    help="directory to create the chromevol input in",
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    "--log_path",
    help="path to log file of the script",
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    "--taxonomic_classification_path",
    help="path to data file with taxonomic classification of members in the counts and tree data",
    type=str,
    required=False,
    default=None
)
def exec_ploidb_pipeline(counts_path: str, tree_path: str, output_dir: str, log_path: str, taxonomic_classification_path: Optional[str]):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    start_time = timer()

    os.makedirs(output_dir, exist_ok=True)
    pipeline = Pipeline(work_dir=output_dir)
    relevant_tree_path = tree_path.replace(".nwk", "_only_with_counts.nwk")
    pipeline.prune_tree_with_counts(counts_path=counts_path, input_tree_path=tree_path,
                                    output_tree_path=relevant_tree_path)

    logger.info(f"selecting the best chromevol model")
    best_model_results_path = pipeline.get_best_model(counts_path=counts_path, tree_path=relevant_tree_path)

    logger.info(f"searching for optimal classification thresholds")
    taxonomic_classification = pd.read_csv(taxonomic_classification_path)
    test_ploidity_classification = pipeline.get_ploidity_classification(counts_path=counts_path,
                                                                        tree_path=relevant_tree_path,
                                                                        full_tree_path=tree_path,
                                                                        model_parameters_path=best_model_results_path,
                                                                        mappings_num=1000,
                                                                        classification_based_on_expectations=False,
                                                                        taxonomic_classification_data=taxonomic_classification)
    test_ploidity_classification.to_csv(f"{output_dir}ploidy.csv")
    pipeline.write_labeled_phyloxml_tree(tree_path=tree_path,
                                         ploidy_classification_data=test_ploidity_classification,
                                         output_path=f"{output_dir}/classified_tree.phyloxml")

    pipeline.write_labeled_newick_tree(tree_path=tree_path,
                                       ploidy_classification_data=test_ploidity_classification,
                                       output_path=f"{output_dir}/classified_tree.newick")

    end_time = timer()
    logger.info(f"overall pipeline duration = {timedelta(seconds=end_time-start_time)}")

if __name__ == '__main__':
    exec_ploidb_pipeline()