import logging
import sys

from ete3 import Tree
from multiprocessing import Pool
from typing import List
import os
import pandas as pd
from functools import partial
import click

logger = logging.getLogger(__name__)

def add_genus_property(tree: Tree):
    for node in tree.get_leaves():
        genus = node.name.split("_")[0]
        node.add_feature(pr_name="genus", pr_value=genus)

def compute_genera_monophyly_classification(genera: List[str], tree: Tree, output_dir: str):
    genus_to_monophyly_status = pd.DataFrame(columns=["genus", "is_monophyletic", "clade_type", "monophyly_violators"])
    genus_to_monophyly_status["genus"] = genera
    genus_to_monophyly_status[["is_monophyletic", "clade_type", "monophyly_violators"]] = genus_to_monophyly_status[["genus"]].apply(lambda record: tree.check_monophyly(values=[record.values[0]], target_attr="genus"), axis=1, result_type="expand")
    output_path = f"{output_dir}/{os.getpid()}.csv"
    genus_to_monophyly_status.to_csv(output_path, index=False)

@click.command()
@click.option(
    "--tree_path",
    help="path to tree file in newick format",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--output_dir",
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
    "--num_cpus",
    help="number of cpus to use",
    type=int,
    required=False,
    default=40,
)
def asses_tree(tree_path: str, output_dir: str, log_path: str, num_cpus: int):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    os.makedirs(output_dir, exist_ok=True)
    tree = Tree(tree_path, format=1)
    add_genus_property(tree)
    genera = list(set([node.genus for node in tree.get_leaves()]))

    logger.info(f"analysis of tree with {len(genera)} genera")
    output_by_batches_dir = f"{output_dir}/genus_to_monophyly_by_batches/"
    os.makedirs(name=output_by_batches_dir, exist_ok=True)
    batch_size = int(len(genera)/num_cpus)+1
    logger.info(f"determined batch size = {batch_size}")
    genera_batches = [genera[i:i+batch_size] for i in range(0, len(genera), batch_size)]
    with Pool(processes=num_cpus) as pool:
        pool.map(partial(compute_genera_monophyly_classification, tree=tree, output_dir=output_dir), genera_batches)
    genus_to_monophyly = pd.concat([pd.read_csv(f"{output_dir}{path}") for path in os.listdir(output_dir) if path.endswith(".csv")])
    genus_to_monophyly.to_csv(f"f{output_dir}/complete_genus_to_monophyly.csv", index=False)

if __name__ == '__main__':
    asses_tree()

