import click
import logging
from pipeline import Pipeline
import sys
import os

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
    "--queue",
    help="queue to submit jobs to",
    type=str,
    required=False,
    default="itaym",
)
@click.option(
    "--max_parallel_jobs",
    help="maximal jobs to submit at the same time from the parent process",
    type=int,
    required=False,
    default=1000,
)
@click.option(
    "--simulations_num",
    help="number of datasets to simulate",
    type=int,
    required=False,
    default=100,
)
@click.option(
    "--trials_num",
    help="number of datasets to attempt to simulate",
    type=int,
    required=False,
    default=1000,
)
def simulate(
    counts_path: str,
    tree_path: str,
    output_dir: str,
    log_path: str,
    parallel: bool,
    ram_per_job: int,
    queue: str,
    max_parallel_jobs: int,
    simulations_num: int,
    trials_num: int,
):

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
    pipeline = Pipeline(work_dir=output_dir,
                        parallel=parallel,
                        ram_per_job=ram_per_job,
                        queue=queue,
                        max_parallel_jobs=max_parallel_jobs)


    logger.info(f"selecting the best chromevol model")
    best_model_results_path = pipeline.get_best_model(
        counts_path=counts_path,
        tree_path=tree_path,
    )

    logger.info(f"simulating data based on selected model = {best_model_results_path}")
    simulations_dirs = pipeline.get_simulations(
                orig_counts_path=counts_path,
                tree_path=tree_path,
                model_parameters_path=best_model_results_path,
                simulations_num=simulations_num,
                trials_num=trials_num,
            )

    print(f"simulations were generated in {os.path.dirname(simulations_dirs[0])}")

    end_time = timer()

    logger.info(f"duration = {timedelta(seconds=end_time-start_time)}")

if __name__ == '__main__':
    simulate()

