import logging
import os
import re
from typing import List, Optional
import getpass
import subprocess
from time import sleep

import numpy as np

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)


class PBSService:
    @staticmethod
    def create_job_file(
        job_path,
        job_name: str,
        job_output_dir: str,
        commands: List[str],
        queue: str = "itaym",
        priority: int = 0,
        cpus_num: int = 1,
        ram_gb_size: int = 4,
    ) -> int:
        os.makedirs(os.path.dirname(job_path), exist_ok=True)
        os.makedirs(os.path.dirname(job_output_dir), exist_ok=True)
        commands_str = "\n".join(commands)
        job_content = f"""# !/bin/bash
    #PBS -S /bin/bash
    #PBS -j oe
    #PBS -r y
    #PBS -q {queue}
    #PBS -p {priority}
    #PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
    #PBS -N {job_name}
    #PBS -e {job_output_dir}
    #PBS -o {job_output_dir}
    #PBS -r y
    #PBS -l select=ncpus={cpus_num}:mem={ram_gb_size}gb
    {commands_str}
    """
        with open(job_path, "w") as outfile:
            outfile.write(job_content)

        return 0

    @staticmethod
    def compute_curr_jobs_num() -> int:
        """
        :return: returns the current number of jobs under the shell username
        """
        username = getpass.getuser()
        proc = subprocess.run(
            f"qselect -u {username} | wc -l",
            shell=True,
            check=True,
            capture_output=True,
        )
        curr_jobs_num = int(proc.stdout)
        return curr_jobs_num

    @staticmethod
    def generate_jobs(
        jobs_commands: List[List[str]],
        work_dir: str,
        output_dir: str,
        ram_per_job_gb: int = 2,
        queue: str = "itaym",
        priority: int = 0,
    ) -> List[str]:
        jobs_paths, job_output_paths = [], []
        for i in range(len(jobs_commands)):
            job_path = f"{work_dir}/{i}.sh"
            job_name = f"{i}.sh"
            job_output_path = f"{output_dir}/{i}.out"
            PBSService.create_job_file(
                job_path=job_path,
                job_name=job_name,
                job_output_dir=job_output_path,
                commands=[os.environ.get("CONDA_ACT_CMD", "")] + jobs_commands[i],
                ram_gb_size=ram_per_job_gb,
                queue=queue,
                priority=priority,
            )
            jobs_paths.append(job_path)
            job_output_paths.append(job_output_path)
        logger.info(f"# jobs to submit = {len(jobs_paths)}")
        return jobs_paths

    @staticmethod
    def retry_memory_failures(jobs_paths: list[str], jobs_output_dir: Optional[str] = None) -> list[str]:
        jobs_ids = []
        mem_re = re.compile("mem=(\d*)gb")
        job_output_path_re = re.compile("#PBS -o (.*?)\n", re.DOTALL)
        for job_path in jobs_paths:
            job_output_path = f"{jobs_output_dir}/{os.path.basename(job_path).replace('.sh', '.out')}"
            if jobs_output_dir is None:
                with open(job_path, "r") as f:
                    job_output_path = job_output_path_re.search(f.read()).group(1)
            if not os.path.exists(job_output_path):
                res = subprocess.check_output(["qsub", f"{job_path}"])
                jobs_ids.append(re.search("(\d+)\.power\d", str(res)).group(1))
            else:
                with open(job_output_path, "r") as f:
                    c = f.read()
                if "PBS: job killed: mem" in c:
                    with open(job_path, "r") as jf:
                        jc = jf.read()
                    mem = int(mem_re.search(jc).group(1))
                    new_mem = mem * 2
                    jc = jc.replace(f"{mem}gb", f"{new_mem}gb")
                    with open(job_path, "w") as jf:
                        jf.write(jc)
                    res = subprocess.check_output(["qsub", f"{job_path}"])
                    jobs_ids.append(re.search("(\d+)\.power\d", str(res)).group(1))
            if os.path.exists(job_output_path):
                os.remove(job_output_path)
        return jobs_ids

    @staticmethod
    def submit_jobs(jobs_paths: List[str], max_parallel_jobs: int = 30, queue: str = "itaym") -> list[str]:
        job_index = 0
        jobs_ids = []
        while job_index < len(jobs_paths):
            while PBSService.compute_curr_jobs_num() > max_parallel_jobs:
                sleep(2 * 60)
            try:
                res = subprocess.check_output(["qsub", "-q", queue, f"{jobs_paths[job_index]}"])
                logger.info(f"job = {jobs_paths[job_index]}, res = {res}")
                jobs_ids.append(re.search("(\d+)\.power\d", str(res)).group(1))
                job_index += 1
            except Exception as e:
                logger.error(f"failed to submit job at index {job_index} due to error {e}")
                exit(1)
            if job_index % 500 == 0:
                logger.info(f"submitted {job_index} jobs thus far")
        return jobs_ids

    @staticmethod
    def wait_for_jobs(jobs_ids: List[str]):
        jobs_complete = np.all([os.system(f"qstat -f {job_id} > /dev/null 2>&1") != 0 for job_id in jobs_ids])
        while not jobs_complete:
            sleep(60)
            jobs_complete = np.all([os.system(f"qstat -f {job_id} > /dev/null 2>&1") != 0 for job_id in jobs_ids])

    @staticmethod
    def execute_job_array(
        work_dir: str,
        output_dir: str,
        jobs_commands: List[List[str]],
        max_parallel_jobs: int = 1900,
        ram_per_job_gb: int = 2,
        queue: str = "itaym",
        priority: int = 0,
    ):
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"# input paths to execute commands on = {len(jobs_commands)}")

        if len(jobs_commands) > 0:
            jobs_paths = PBSService.generate_jobs(
                jobs_commands=jobs_commands,
                work_dir=work_dir,
                output_dir=output_dir,
                ram_per_job_gb=ram_per_job_gb,
                queue=queue,
                priority=priority,
            )
            jobs_ids = PBSService.submit_jobs(jobs_paths=jobs_paths, max_parallel_jobs=max_parallel_jobs, queue=queue)
            done = False
            while not done:
                PBSService.wait_for_jobs(jobs_ids=jobs_ids)
                job_ids = PBSService.retry_memory_failures(jobs_paths=jobs_paths, jobs_output_dir=output_dir)
                done = len(job_ids) == 0

        # # remove work dir
        # shutil.rmtree(work_dir, ignore_errors=True)
