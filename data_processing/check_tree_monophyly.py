import os
import sys
from typing import Tuple

import logging
import numpy as np
from ete3 import Tree
import pandas as pd
import click

logger = logging.getLogger(__name__)


def add_group_by_property(
    tree: Tree, classification_data: pd.DataFrame, class_name: str
):
    taxon_to_class = classification_data.set_index("taxon")[class_name].to_dict()
    for node in tree.get_leaves():
        node_class = taxon_to_class.get(node.name, np.nan)
        if class_name == "genus" and pd.isna(node_class):
            node_class = node.name.split(" ")[0]
        node.add_feature(pr_name=class_name, pr_value=node_class)


def get_largest_monophyletic_group(
    root: Tree, property_value: str, property_name: str, full_group_size: int
) -> Tuple[
    Tree, float, bool, pd.DataFrame
]:  # returns: root of subtree, number of members in the subtree, indicator if the subtree is larger than others, % members covered in the group, % members out of subtree
    ancestor_to_stats = pd.DataFrame(
        columns=[
            "node",
            "num_members",
            "size_subtree",
            "total_members_coverage",
            "subtree_members_coverage",
        ]
    )

    if root.is_leaf():
        return root, 0, False, ancestor_to_stats

    q = [root]
    monophyletic_ancestors_to_size = dict()
    while len(q) > 0:
        node = q.pop(0)
        property_values = [leaf.__dict__[property_name] for leaf in node.get_leaves()]
        node_stats = {
            "node": [node.name],
            "num_members": [property_values.count(property_value)],
            "size_subtree": [len(node.get_leaves())],
            "total_members_coverage": [
                property_values.count(
                    property_value
                )  # measure of how many members are covered by the subtree of out of all the members
                / full_group_size
            ],
            "subtree_members_coverage": [
                (property_values.count(property_value) / len(node.get_leaves()))
            ],  # measure of how many of the leaves in the subtree are members
        }
        ancestor_to_stats = pd.concat(
            [ancestor_to_stats, pd.DataFrame.from_dict(node_stats, orient="columns")]
        )
        if len(set(property_values)) == 1 and property_values[0] == property_value:
            monophyletic_ancestors_to_size[node] = len(node.get_leaves())
            if node_stats["total_members_coverage"][0] == 1:
                break
        for child in node.get_children():
            q.append(child)
    best_node = max(
        monophyletic_ancestors_to_size, key=monophyletic_ancestors_to_size.get
    )
    is_larger_than_rest = (
        len(
            [
                n
                for n in monophyletic_ancestors_to_size
                if monophyletic_ancestors_to_size[n]
                == monophyletic_ancestors_to_size[best_node]
            ]
        )
        == 1
    )
    return (
        best_node,
        monophyletic_ancestors_to_size[best_node],
        is_larger_than_rest,
        ancestor_to_stats,
    )


def check_monophyly(
    record: pd.Series, tree: Tree, property_name: str, ancestor_to_stats_path: str
) -> Tuple[bool, str, int, str, str, float, float]:
    group = record[property_name]
    group_leaves = [
        leaf for leaf in tree.get_leaves() if leaf.__dict__[property_name] == group
    ]
    group_size = len(group_leaves)
    if len(group_leaves) < 2:
        return (
            True,
            "monophyletic",
            group_size,
            "",
            "".join([l.name for l in group_leaves]),
            1,
            0,
        )
    is_monophyletic, clade_type, monophyly_violators = tree.check_monophyly(
        values=[group], target_attr=property_name
    )
    monophyly_violators_names = (
        ",".join([node.name for node in monophyly_violators if node.is_leaf()])
        if len(monophyly_violators) > 0
        else np.nan
    )
    lca = tree.get_common_ancestor(group_leaves)
    (
        largest_monophyly_root,
        largest_monophyly_size,
        is_larger_than_rest,
        ancestor_to_stats,
    ) = get_largest_monophyletic_group(
        root=lca,
        property_value=group,
        property_name=property_name,
        full_group_size=group_size,
    )
    largest_monophyletic_group_names = largest_monophyly_root.get_leaf_names()
    fraction_largest_monophyletic_group = float(largest_monophyly_size) / float(
        len(group_leaves)
    )
    fraction_of_invaders = len(monophyly_violators) / (
        len(group_leaves) + len(monophyly_violators)
    )
    ancestor_to_stats.to_csv(ancestor_to_stats_path, index=False)

    return (
        bool(is_monophyletic),
        str(clade_type),
        group_size,
        monophyly_violators_names,
        largest_monophyletic_group_names,
        fraction_largest_monophyletic_group,
        fraction_of_invaders,
    )


@click.command()
@click.option(
    "--group_path",
    help="path csv with the groups values to compute monophyly for",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--group_by",
    help="item to group by (either genus of family)",
    type=click.Choice(["genus", "family"]),
    required=True,
)
@click.option(
    "--tree_path",
    help="path to tree file in newick format",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--classification_path",
    help="path with mapping of each taxon to genus/ family",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--output_path",
    help="path to write output to",
    type=click.Path(exists=False),
    required=True,
)
def compute_genera_monophyly_classification(
    group_path: str,
    group_by: str,
    classification_path: str,
    tree_path: str,
    output_path: str,
):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    tree = Tree(tree_path, format=1)
    classification_data = pd.read_csv(classification_path)
    add_group_by_property(
        tree, classification_data=classification_data, class_name=group_by
    )
    logger.info(f"parsed input tree and groups classification per leaf")

    groups = pd.read_csv(group_path)["0"].tolist()
    logger.info(f"checking monophyly status for {len(groups)} groups")

    group_to_monophyly_status = pd.DataFrame(
        columns=[group_by, "is_monophyletic", "clade_type", "monophyly_violators"]
    )
    group_to_monophyly_status[group_by] = groups
    group_to_monophyly_status[
        [
            "is_monophyletic",
            "clade_type",
            "group_size",
            "monophyly_violators",
            "largest_monophyly_members",
            "largest_monophyletic_fraction",
            "fraction_of_invaders",
        ]
    ] = group_to_monophyly_status.apply(
        lambda record: check_monophyly(
            record=record,
            tree=tree,
            property_name=group_by,
            ancestor_to_stats_path=f"{os.path.dirname(output_path)}/{group_by}_{record[group_by]}_ancestry_stats.csv",
        ),
        axis=1,
        result_type="expand",
    )
    group_to_monophyly_status.to_csv(output_path, index=False)


if __name__ == "__main__":
    compute_genera_monophyly_classification()
