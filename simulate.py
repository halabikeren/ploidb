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
    "--work_dir",
    help="directory to create the chromevol input in",
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    "--simulations_dir",
    help="directory to create the simulations in",
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
@click.option(
    "--use_model_selection",
    help="indicator of weather model selection should be used",
    type=bool,
    required=False,
    default=True,
)
def simulate(
    counts_path: str,
    tree_path: str,
    work_dir: str,
    simulations_dir: str,
    log_path: str,
    parallel: bool,
    ram_per_job: int,
    queue: str,
    max_parallel_jobs: int,
    simulations_num: int,
    trials_num: int,
    use_model_selection: bool,
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

    os.makedirs(work_dir, exist_ok=True)
    pipeline = Pipeline(
        work_dir=work_dir, parallel=parallel, ram_per_job=ram_per_job, queue=queue, max_parallel_jobs=max_parallel_jobs
    )

    logger.info(f"selecting the best chromevol model")
    model_selection_dir = f"{work_dir}/model_selection/"
    if os.path.exists(model_selection_dir):
        for model_name in os.listdir(model_selection_dir):
            out_path = f"{model_selection_dir}{model_name}/parsed_output.json"
            if os.path.exists(out_path):
                os.remove(out_path)

    model_to_weight = pipeline.get_model_weights(
        counts_path=counts_path,
        tree_path=tree_path,
        use_model_selection=use_model_selection,
    )
    best_model_results_path = list(model_to_weight.keys())[0]

    os.makedirs(simulations_dir, exist_ok=True)
    logger.info(f"simulating data based on selected model = {best_model_results_path}")
    simulations_dirs = pipeline.get_simulations(
        simulations_dir=simulations_dir,
        orig_counts_path=counts_path,
        tree_path=tree_path,
        model_parameters_path=best_model_results_path,
        simulations_num=simulations_num,
        trials_num=trials_num,
    )

    print(f"simulations were generated in {simulations_dir}")

    end_time = timer()

    logger.info(f"duration = {timedelta(seconds=end_time-start_time)}")


if __name__ == "__main__":
    simulate()
