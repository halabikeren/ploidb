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
@click.option(
    "--parallel",
    help="indicator weather to run the pipeline in parallel (1) with one idle parent job or sequentially",
    type=bool,
    required=False,
    default=False
)
@click.option(
    "--diploidy_threshold",
    help="threshold between 0 and 1 for the frequency of polyploidy support across mappings for taxa to be deemed as diploids",
    type= click.FloatRange(min=0, max=1),
    required=False,
    default=0.1
)
@click.option(
    "--polyploidy_threshold",
    help="threshold between 0 and 1 for the frequency of polyploidy support across mappings for taxa to be deemed as polyploids",
    type= click.FloatRange(min=0, max=1),
    required=False,
    default=0.9
)
def exec_ploidb_pipeline(counts_path: str,
                         tree_path: str,
                         output_dir: str,
                         log_path: str,
                         taxonomic_classification_path: Optional[str],
                         parallel: bool,
                         diploidy_threshold: float,
                         polyploidy_threshold: float):

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
    best_model_results_path = pipeline.get_best_model(counts_path=counts_path, tree_path=relevant_tree_path, parallel=parallel)

    logger.info(f"searching for optimal classification thresholds")
    taxonomic_classification = pd.read_csv(taxonomic_classification_path) if taxonomic_classification_path is not None else None
    test_ploidity_classification = pipeline.get_ploidity_classification(counts_path=counts_path,
                                                                        tree_path=relevant_tree_path,
                                                                        full_tree_path=tree_path,
                                                                        model_parameters_path=best_model_results_path,
                                                                        mappings_num=1000,
                                                                        taxonomic_classification_data=taxonomic_classification,
                                                                        parallel=parallel,
                                                                        diploidity_threshold=diploidy_threshold,
                                                                        polyploidity_threshold=polyploidy_threshold,
                                                                        optimize_thresholds=True)
    test_ploidity_classification.to_csv(f"{output_dir}ploidy.csv", index=False)
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