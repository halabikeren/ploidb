import os
import sys

import click
import logging
import pandas as pd
from ete3 import Tree
from typing import Optional

logger = logging.getLogger(__name__)


def write_class_tree(full_tree: Tree, class_members: list[str], output_path: str):
    if not os.path.exists(output_path):
        logger.info(f"writing genus tree of size {len(class_members)} to {output_path}")
        if len(class_members) >= 5:
            tax_class_tree = full_tree.copy()
            tax_class_tree.prune(class_members, preserve_branch_length=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tax_class_tree.write(outfile=output_path, format=1)


@click.command()
@click.option(
    "--partition_data_path",
    help="path to csv file with the name of the internal node that was selected as the optimal mrca of each "
         "classification group. If not provided, simple tree pruning will be applied",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=False,
    default=None
)
@click.option(
    "--taxonomic_classification_path",
    help="path to csv file with classification of each tree species to the taxonomic class of interest",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--query_field_name",
    help="name of class field from the taxonomic_classification_path search for taxonomic class by",
    type=str,
    required=False,
    default="resolved_species_name",
)
@click.option(
    "--class_field_name",
    help="name of class field from the taxonomic_classification_path to prune data by",
    type=click.Choice(["genus", "family"]),
    required=False,
    default="genus",
)
@click.option(
    "--tree_path",
    help="path to the processed tree file with the resolved names",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--output_dir",
    help="directory to write the output trees, following their extraction from the input tree, to",
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    "--log_path",
    help="path to logger",
    type=click.Path(exists=False),
    required=True,
)
def divide_tree(
    taxonomic_classification_path: str,
    query_field_name: str,
    class_field_name: str,
    tree_path: str,
    output_dir: str,
    log_path: str,
    partition_data_path: Optional[str]
):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    tree = Tree(tree_path, format=1)
    logger.info(f"the full tree consists of {len(tree.get_leaf_names())} species")
    taxonomic_classification = pd.read_csv(taxonomic_classification_path)[[query_field_name, class_field_name]].dropna()
    taxonomic_classification = taxonomic_classification.loc[
        taxonomic_classification[query_field_name].isin(tree.get_leaf_names())
    ]
    logger.info(f"out of these, classification is available for {taxonomic_classification.shape[0]} species")
    logger.info(f"will now write {class_field_name} trees to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    if partition_data_path:
        partition_data = pd.read_csv(partition_data_path)
        for i, row in partition_data.iterrows():
            group = row[class_field_name]
            mrca_name = row["node"]
            mrca = tree.search_nodes(name=mrca_name)[0].copy()
            members = [l.name for l in mrca.get_leaves() if l.__dict__[class_field_name].lower() == group.lower()]
            if len(members) != row["num_members"]:
                print(f"for group {group}, the number of leaves is {len(members)} while there should be {row.num_members}")
            mrca.prune(members, preserve_branch_length=True)
            os.makedirs(f"{output_dir}{group}/", exist_ok=True)
            mrca.write(outfile=f"{output_dir}{group}/tree.nwk")
    else:
        taxonomic_classification.groupby(class_field_name).apply(
            lambda g: write_class_tree(
                full_tree=tree,
                class_members=g[query_field_name].values,
                output_path=f"{output_dir}{g[class_field_name].values[0]}/tree.nwk",
            )
        )


if __name__ == "__main__":
    divide_tree()
