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
    default=None,
)
@click.option(
    "--ploidy_classification_path",
    help="path to write the ploidy classification to",
    type=str,
    required=False,
    default=None,
)
@click.option(
    "--parallel",
    help="indicator weather to run the pipeline in parallel (1) with one idle parent job or sequentially",
    type=bool,
    required=False,
    default=False,
)
@click.option(
    "--ram_per_job",
    help="memory size per job to parallelize on",
    type=int,
    required=False,
    default=1,
)
@click.option(
    "--optimize_thresholds",
    help="indicator weather thresholds should be optimized based on simulations",
    type=bool,
    required=False,
    default=False,  # change to false for unfinished jobs
)
@click.option(
    "--diploidy_threshold",
    help="threshold between 0 and 1 for the frequency of polyploidy support across mappings for taxa to be deemed as diploids",
    type=click.FloatRange(min=0, max=1),
    required=False,
    default=0.25,
)
@click.option(
    "--polyploidy_threshold",
    help="threshold between 0 and 1 for the frequency of polyploidy support across mappings for taxa to be deemed as polyploids",
    type=click.FloatRange(min=0, max=1),
    required=False,
    default=0.75,
)
@click.option(
    "--queue",
    help="queue to submit jobs to",
    type=str,
    required=False,
    default="itaym",
)
@click.option(
    "--debug_sim_num",
    help="indicator weather simulations based threshold optimization should be debugged or not",
    type=bool,
    required=False,
    default=False,
)
@click.option(
    "--max_parallel_jobs",
    help="maximal jobs to submit at the same time from the parent process",
    type=int,
    required=False,
    default=1000,
)
@click.option(
    "--allow_base_num_parameter",
    help="indicator if we allow the selected model to include base number parameter or not",
    type=bool,
    required=False,
    default=True,
)
@click.option(
    "--use_model_selection",
    help="indicator if we allow the selected model to include base number parameter or not",
    type=bool,
    required=False,
    default=True,
)
def exec_ploidb_pipeline(
    counts_path: str,
    tree_path: str,
    output_dir: str,
    log_path: str,
    taxonomic_classification_path: Optional[str],
    parallel: bool,
    ram_per_job: int,
    optimize_thresholds: bool,
    diploidy_threshold: float,
    polyploidy_threshold: float,
    queue: str,
    debug_sim_num: bool,
    max_parallel_jobs: int,
    ploidy_classification_path: str,
    allow_base_num_parameter: bool,
    use_model_selection: bool,
):

    if ploidy_classification_path is None:
        ploidy_classification_path = (
            f"{output_dir}/ploidy{'_without_base_num' if not allow_base_num_parameter else ''}.csv"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    res = os.system(f"dos2unix {counts_path}")
    res = os.system(f"dos2unix {tree_path}")

    start_time = timer()

    os.makedirs(output_dir, exist_ok=True)
    pipeline = Pipeline(
        work_dir=output_dir,
        parallel=parallel,
        ram_per_job=ram_per_job,
        queue=queue,
        max_parallel_jobs=max_parallel_jobs,
    )

    logger.info(f"selecting the best chromevol model")
    model_path_to_weight = pipeline.get_model_weights(
        counts_path=counts_path,
        tree_path=tree_path,
        allow_base_num_parameter=allow_base_num_parameter,
        use_model_selection=use_model_selection,
    )

    if optimize_thresholds:
        logger.info(f"searching for optimal classification thresholds")
    else:
        logger.info(
            f"determining ploidy level based on the fixed diploidy and polyploidy thresholds {diploidy_threshold} and {polyploidy_threshold}"
        )
    taxonomic_classification = (
        pd.read_csv(taxonomic_classification_path) if taxonomic_classification_path is not None else None
    )

    test_ploidy_classification = pipeline.get_ploidy_classification(
        counts_path=counts_path,
        tree_path=tree_path,
        weighted_models_parameters_paths=model_path_to_weight,
        mappings_num=1000,
        taxonomic_classification_data=taxonomic_classification,
        diploidy_threshold=diploidy_threshold,
        polyploidy_threshold=polyploidy_threshold,
        optimize_thresholds=optimize_thresholds,
        debug=debug_sim_num,
        use_model_selection=use_model_selection,
    )
    test_ploidy_classification.to_csv(ploidy_classification_path, index=False)
    pipeline.write_labeled_phyloxml_tree(
        tree_path=tree_path,
        ploidy_classification_data=test_ploidy_classification,
        output_path=f"{os.path.dirname(ploidy_classification_path)}/classified_tree.phyloxml",
    )

    pipeline.write_labeled_newick_tree(
        tree_path=tree_path,
        ploidy_classification_data=test_ploidy_classification,
        output_path=f"{os.path.dirname(ploidy_classification_path)}/classified_tree.newick",
    )

    end_time = timer()
    logger.info(f"overall pipeline duration = {timedelta(seconds=end_time-start_time)}")


if __name__ == "__main__":
    exec_ploidb_pipeline()
