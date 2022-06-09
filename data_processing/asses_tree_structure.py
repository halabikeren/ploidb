import logging
import sys
from ete3 import Tree
import os
import pandas as pd
import click

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services.pbs_service import PBSService
from data_processing.check_tree_monophyly import add_genus_property

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--tree_path",
    help="path to tree file in newick format",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--work_dir",
    help="directory to which the output should be written",
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    "--log_path",
    help="path to the logging data",
    type=click.Path(exists=False),
    required=True,
)
def asses_tree(tree_path: str, work_dir: str, log_path: str):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    os.makedirs(work_dir, exist_ok=True)
    output_dir = f"{work_dir}/monophyly_status/"
    os.makedirs(output_dir, exist_ok=True)
    genera_batches_dir = f"{work_dir}/genera/"
    os.makedirs(genera_batches_dir, exist_ok=True)

    tree = Tree(tree_path, format=1)
    add_genus_property(tree)
    genera = list(set([node.genus for node in tree.get_leaves()]))

    logger.info(f"analysis of tree with {len(genera)} genera")
    batch_size = int(len(genera)/50)+1
    logger.info(f"determined batch size = {batch_size}")
    genera_batches = [genera[i:i+batch_size] for i in range(0, len(genera), batch_size)]
    jobs_commands = []
    for i in range(len(genera_batches)):
        genera_path = f"{genera_batches_dir}batch_{i}_genera.csv"
        pd.Series(genera_batches[i]).to_csv(genera_path, index=False)
        output_path = f"{output_dir}/batch_{i}_monophyly_status.csv"
        if not os.path.exists(output_path):
            batch_commands = [f"cd {os.path.dirname(__file__)}", f"python check_tree_monophyly.py --genera_path={genera_path} --tree_path={tree_path} --output_path={output_path}"]
            jobs_commands.append(batch_commands)
    if len(jobs_commands) > 0:
        logger.info(f"submitting {len(jobs_commands)} jobs for branches of genera")
        PBSService.execute_job_array(work_dir=f"{work_dir}/jobs/",
                                     output_dir=f"{work_dir}/jobs_output/",
                                     jobs_commands=jobs_commands,)

    logger.info(f"processing of genera monophyly status is complete, will now merge results")
    genus_to_monophyly = pd.concat([pd.read_csv(f"{output_dir}{path}") for path in os.listdir(output_dir) if path.endswith(".csv")])
    genus_to_monophyly.to_csv(tree_path.replace(".tre", "_complete_genus_to_monophyly.csv"), index=False)

if __name__ == '__main__':
    asses_tree()

