import logging
import sys
from ete3 import Tree
import os
import pandas as pd
import click

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services.pbs_service import PBSService
from data_processing.check_tree_monophyly import add_group_by_property

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--tree_path",
    help="path to tree file in newick format",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--group_by",
    help="taxonomy level to group by",
    type=click.Choice(["genus", "family"]),
    required=True,
)
@click.option(
    "--classification_path",
    help="path to dataframe classifying each leaf to genus/ family",
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
@click.option(
    "--output_path",
    help="path of merged result",
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    "--queue",
    help="queue to submit jobs to",
    type=str,
    required=False,
    default="itaym",
)
@click.option(
    "--batch_num",
    help="number of batches to divide instances to",
    type=int,
    required=False,
    default=400,
)
@click.option(
    "--job_num_limit",
    help="number of job to submit at the same time",
    type=int,
    required=False,
    default=100,
)
@click.option(
    "--priority", help="jobs priority", type=int, required=False, default=-1,
)
def asses_tree(
    tree_path: str,
    group_by: str,
    classification_path: str,
    work_dir: str,
    log_path: str,
    batch_num: int,
    output_path: str,
    queue: str,
    job_num_limit: int,
    priority: int,
):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    os.makedirs(work_dir, exist_ok=True)
    output_dir = f"{work_dir}/monophyly_status/"
    os.makedirs(output_dir, exist_ok=True)
    genera_batches_dir = f"{work_dir}/by_{group_by}/"
    os.makedirs(genera_batches_dir, exist_ok=True)

    classification_data = pd.read_csv(classification_path)
    tree = Tree(tree_path, format=1)
    add_group_by_property(
        tree=tree, classification_data=classification_data, class_name=group_by,
    )
    groups = list(set([leaf.__dict__[group_by] for leaf in tree.get_leaves()]))
    logger.info(f"analysis of tree with {len(groups)} {group_by} unique values")

    batch_size = int(len(groups) / batch_num) + 1
    logger.info(f"determined batch size = {batch_size}")

    genera_batches = [
        groups[i : i + batch_size] for i in range(0, len(groups), batch_size)
    ]
    logger.info(f"num batches = {len(genera_batches):,}")

    jobs_commands = []
    for i in range(len(genera_batches)):
        group_path = f"{genera_batches_dir}batch_{i}_{group_by}.csv"
        pd.Series(genera_batches[i]).to_csv(group_path, index=False)
        batch_output_path = f"{output_dir}/batch_{i}_monophyly_status.csv"
        if not os.path.exists(batch_output_path):
            batch_commands = [
                f"cd {os.path.dirname(__file__)}",
                f"python check_tree_monophyly.py --group_path={group_path} --classification_path={classification_path} --group_by={group_by} --tree_path={tree_path} --output_path={batch_output_path}",
            ]
            jobs_commands.append(batch_commands)
    if len(jobs_commands) > 0:
        logger.info(f"submitting {len(jobs_commands)} jobs for batches of {group_by}")
        PBSService.execute_job_array(
            work_dir=f"{work_dir}/jobs/",
            output_dir=f"{work_dir}/jobs_output/",
            jobs_commands=jobs_commands,
            queue=queue,
            ram_per_job_gb=1,
            max_parallel_jobs=job_num_limit,
            priority=priority,
        )

    logger.info(
        f"processing of genera monophyly status is complete, will now merge results"
    )
    status_paths = [
        f"{output_dir}{path}"
        for path in os.listdir(output_dir)
        if path.endswith("_monophyly_status.csv")
    ]
    anc_paths = [f"{output_dir}{path}" for path in os.listdir(output_dir) if "batch" not in path]
    print(f"# paths to concat for monophyly status analysis = {len(status_paths)}")
    group_to_monophyly = pd.concat([pd.read_csv(path) for path in status_paths])
    group_to_monophyly.drop_duplicates(inplace=True)
    group_to_monophyly.to_csv(output_path, index=False)

    print(f"# paths to concat for best ancestor to {group_by} = {len(anc_paths)}")
    anc_dfs = []
    for path in anc_paths:
        df = pd.read_csv(path)
        df["genus"] = os.path.basename(path).split("_")[1]
        anc_dfs.append(df)
    anc_df = pd.concat(anc_dfs)
    anc_df["mrca_score"] = (anc_df.total_members_coverage + anc_df.subtree_members_coverage) / 2
    anc_df = anc_df.sort_values(["genus", "mrca_score", "num_members"], ascending=[True, False, False])
    anc_df.to_csv(os.path.dirname(output_path) + f"/{group_by}_to_ancestor", index=False)

if __name__ == "__main__":
    asses_tree()
