import re
import sys
from typing import Tuple

import logging
import numpy as np
from ete3 import Tree
import pandas as pd
import click

logger = logging.getLogger(__name__)

def add_genus_property(tree: Tree):
    for node in tree.get_leaves():
        genus = re.split("_|\s", node.name)[0]
        node.add_feature(pr_name="genus", pr_value=genus)

def get_largest_monophyletic_group(root: Tree, property_value: str, property_name: str = "genus") -> Tuple[Tree, float, bool]:
    if root.is_leaf():
        return root, 0, False

    q = [root]
    monophyletic_ancestors_to_size = dict()
    while len(q) > 0:
        node = q.pop(0)
        property_values = list(set([leaf.__dict__[property_name] for leaf in node.get_leaves()]))
        if len(property_values) == 1 and property_values[0] == property_value:
            monophyletic_ancestors_to_size[node] = len(node.get_leaves())
        for child in node.get_children():
            q.append(child)
    best_node = max(monophyletic_ancestors_to_size, key=monophyletic_ancestors_to_size.get)
    is_larger_than_rest = len([n for n in monophyletic_ancestors_to_size if monophyletic_ancestors_to_size[n] == monophyletic_ancestors_to_size[best_node]]) == 1
    return best_node, monophyletic_ancestors_to_size[best_node], is_larger_than_rest


def check_monophyly(record: pd.Series, tree: Tree, property_name: str = "genus") -> Tuple[bool, str, str, str, float, float]:
    genus = record.genus
    genus_leaves = [leaf for leaf in tree.get_leaves() if leaf.genus == genus]
    if len(genus_leaves) < 2:
        return True, "monophyletic", "", "".join([l.name for l in genus_leaves]), 1, 0
    is_monophyletic, clade_type, monophyly_violators = tree.check_monophyly(values=[genus], target_attr=property_name)
    monophyly_violators_names = ",".join([node.name for node in monophyly_violators if node.is_leaf()]) if len(monophyly_violators) > 0 else np.nan
    lca = tree.get_common_ancestor(genus_leaves)
    largest_monophyly_root, largest_monophyly_size, is_larger_than_rest = get_largest_monophyletic_group(root=lca, property_value=genus, property_name="genus")
    largest_monophyletic_group_names = largest_monophyly_root.get_leaf_names()
    fraction_largest_monophyletic_group = float(largest_monophyly_size)/float(len(genus_leaves))
    fraction_of_invaders = len(monophyly_violators) / (len(genus_leaves)+len(monophyly_violators))
    return bool(is_monophyletic), str(clade_type), monophyly_violators_names, largest_monophyletic_group_names, fraction_largest_monophyletic_group, fraction_of_invaders

@click.command()
@click.option(
    "--genera_path",
    help="path csv with the genera to compute monophyly for",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--tree_path",
    help="path to tree file in newick format",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--output_path",
    help="path to write output to",
    type=click.Path(exists=False),
    required=True,
)
def compute_genera_monophyly_classification(genera_path: str, tree_path: str, output_path: str):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    tree = Tree(tree_path, format=1)
    add_genus_property(tree)
    logger.info(f"parsed input tree and genera classification per leaf")

    genera = pd.read_csv(genera_path)["0"].tolist()
    logger.info(f"checking monophyly status for {len(genera)} genera")

    genus_to_monophyly_status = pd.DataFrame(columns=["genus", "is_monophyletic", "clade_type", "monophyly_violators"])
    genus_to_monophyly_status["genus"] = genera
    genus_to_monophyly_status[["is_monophyletic", "clade_type", "monophyly_violators", "largest_monophyly_members", "largest_monophyletic_fraction", "fraction_of_invaders"]] = genus_to_monophyly_status.apply(lambda record: check_monophyly(record=record, tree=tree), axis=1, result_type="expand")
    genus_to_monophyly_status.to_csv(output_path, index=False)

if __name__ == '__main__':
    compute_genera_monophyly_classification()