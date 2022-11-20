import sys
import click
import logging
import pickle
from pbs_service import PBSService
from timeit import default_timer as timer
from datetime import timedelta

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--jobs_list_path",
    help="path to pickle file with job paths to run",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--log_path",
    help="path to the log file",
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    "--max_parallel_jobs",
    help="maximal number of jobs to exist at the same time",
    type=int,
    required=False,
    default=1900,
)
@click.option("--queue", help="queue to submit jobs to", type=str, required=False, default="itaym")
def schedule_jobs(jobs_list_path: str, log_path: str, max_parallel_jobs: int, queue: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )
    start_time = timer()

    with open(jobs_list_path, "rb") as f:
        jobs_paths = pickle.load(f)

    jobs_ids = PBSService.submit_jobs(jobs_paths=jobs_paths, max_parallel_jobs=max_parallel_jobs, queue=queue)
    done = False
    while not done:
        PBSService.wait_for_jobs(jobs_ids=jobs_ids)
        job_ids = PBSService.retry_memory_failures(jobs_paths=jobs_paths)
        done = len(job_ids) == 0

    end_time = timer()
    logger.info(f"overall scheduler duration = {timedelta(seconds=end_time-start_time)}")


if __name__ == "__main__":
    schedule_jobs()
