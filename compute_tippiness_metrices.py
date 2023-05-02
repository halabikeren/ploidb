import pandas as pd
import numpy as np
import os
from Bio import SeqIO
import shutil
import pickle
import sys
import re
import click
from zipfile import ZipFile

sys.path.append("/groups/itay_mayrose/halabikeren/tmp/ploidb/services/")
from pbs_service import PBSService
from ete3 import Tree

base = "ott"

chromevol_genera_dir = (
    "/groups/itay_mayrose/halabikeren/PloiDB/chromevol/with_model_weighting/by_genus_on_unresolved_ALLMB_and_unresolved_ccdb/"
    if base == "allmb"
    else "/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/one_two_tree/genus/ploidb_pipeline_with_model_weighting/"
)
poc_chromevol_dir = (
    f"/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/poc_from_sim_to_sm/{base}_based_ploidb_pipeline/"
)

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=10)


# compute relative age
def get_relative_age(record: pd.Series, tree_height: float) -> float:
    age = record.age
    dataset = record.dataset
    try:
        relative_age = age / tree_height
        return relative_age
    except Exception as e:
        print(e)
        return np.nan


def get_frac_external_polyploidizations(df: pd.DataFrame):
    df["is_event_poyploidization"] = df.event_type.isin(["DUPLICATION", "BASE-NUMBER", "DEMI-DUPLICATION"])
    num_polyploidizations = df.loc[df.is_event_poyploidization].shape[0]
    if num_polyploidizations == 0:
        return 0
    num_external_polyploidizations = df.loc[df.is_event_poyploidization & df.is_child_external].shape[0]
    return num_external_polyploidizations / num_polyploidizations


def get_polyploidization_mean_age(df: pd.DataFrame):
    df["is_event_poyploidization"] = df.event_type.isin(["DUPLICATION", "BASE-NUMBER", "DEMI-DUPLICATION"])
    if df.loc[df.is_event_poyploidization].shape[0] == 0:
        return np.nan
    age_col = "age" if base == "allmb" else "relative_age"
    mean_polyploidization_age = df.loc[df.is_event_poyploidization].age.mean()
    return mean_polyploidization_age


def get_mean_bl(df: pd.DataFrame, tree_height: float) -> float:
    leaves_df = df.loc[
        (df.is_child_external) & (df.event_type.isin(["DUPLICATION", "BASE-NUMBER", "DEMI-DUPLICATION"]))
    ]
    bls = leaves_df.branch_length.tolist()  # the age of leaves is equivalent to their branch length
    if len(bls) == 0:
        return np.nan
    return np.mean(bls) / tree_height


def get_mean_poly_clade_size(df: pd.DataFrame, tree: Tree, use_relative: bool = False) -> float:
    try:
        df = df.sort_values("age", ascending=False)
        polyploids = (
            df.loc[df.event_type.isin(["DUPLICATION", "BASE-NUMBER", "DEMI-DUPLICATION"])]
            .branch_child_name.unique()
            .tolist()
        )
        clades = []
        i = 0
        while len(polyploids) > 0:
            polyploid = polyploids[0]
            subtree = tree.search_nodes(name=polyploid)[0]
            subtree_members = set([n.name for n in subtree.traverse()])
            to_remove = []
            for poly in polyploids:
                if poly != polyploid and poly in subtree_members:
                    to_remove.append(poly)
            polyploids = [p for p in polyploids if not p in to_remove]
            clades.append(subtree)
            polyploids.remove(polyploid)
        clade_sizes = [len(clade.get_leaves()) for clade in clades]
        mean_clade_size = np.mean(clade_sizes)
        if use_relative:
            mean_clade_size = mean_clade_size / len(tree.get_leaves())
        return mean_clade_size
    except Exception as e:
        print(e)
        return np.nan


def get_ml_tree(dataset: str) -> Tree:
    orig_tree = f"{chromevol_genera_dir}{dataset}/tree.nwk"
    orig_tree = Tree(orig_tree)
    i = 0
    while i < 100:
        tree_path = f"{chromevol_genera_dir}{dataset}/chromevol/100_simulations/{i}/simulatedDataAncestors.tree"
        if os.path.exists(tree_path):
            tree = Tree(tree_path, format=1)
            scaling_factor = np.sum([n.dist for n in orig_tree.traverse()]) / np.sum([n.dist for n in tree.traverse()])
            for node in tree.traverse():
                node.dist = node.dist * scaling_factor
                node.name = "-".join(node.name.split("-")[:-1])
            return tree
        i += 1
    print(f"no tree was found in {chromevol_genera_dir}{dataset}/chromevol/100_simulations/")
    return np.nan


@click.command()
@click.option(
    "--dataset",
    help="name of genus",
    type=str,
    required=True,
)
def process_data(dataset: str):

    raw_simulations_data_path = f"/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/poc_from_sim_to_sm/{base}_based_data_by_dataset/dataset_{dataset}_simulations_raw_data.csv"
    raw_mappings_data_path = f"/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/poc_from_sim_to_sm/{base}_based_data_by_dataset/dataset_{dataset}_parametric_boostrapping_raw_data.csv"

    processed_simulations_data_path = f"/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/poc_from_sim_to_sm/{base}_based_data_by_dataset/dataset_{dataset}_simulations_processed_data.csv"
    processed_mappings_data_path = f"/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/poc_from_sim_to_sm/{base}_based_data_by_dataset/dataset_{dataset}_parametric_bootstrapping_processed_data.csv"

    simulations = pd.read_csv(raw_simulations_data_path)
    mappings = pd.read_csv(raw_mappings_data_path)

    dataset = simulations.dataset.unique().tolist()[0]
    tree = get_ml_tree(dataset)
    tree_height = tree.get_distance(tree.get_leaves()[0])
    external_branches = set(tree.get_leaf_names())

    node_to_bl = {node.name: node.dist for node in tree.traverse() if node.name != ""}
    bls_data = (
        pd.DataFrame.from_dict(node_to_bl, orient="index")
        .reset_index()
        .rename(columns={"index": "branch_child_name", 0: "branch_length"})
    )

    simulations["is_child_external"] = simulations.apply(lambda rec: rec.branch_child_name in external_branches, axis=1)
    mappings["is_child_external"] = mappings.apply(lambda rec: rec.branch_child_name in external_branches, axis=1)

    # fill branch lengths data
    mappings = mappings.merge(bls_data, on=["branch_child_name"], how="left")
    simulations = simulations.merge(bls_data, on=["branch_child_name"], how="left")

    assert "is_child_external" in simulations.columns
    assert "is_child_external" in mappings.columns
    assert "branch_length" in simulations.columns
    assert "branch_length" in mappings.columns

    mappings["relative_age"] = mappings.parallel_apply(
        lambda rec: get_relative_age(rec, tree_height=tree_height), axis=1
    )
    simulations["relative_age"] = simulations.parallel_apply(
        lambda rec: get_relative_age(rec, tree_height=tree_height), axis=1
    )

    simulations = simulations.loc[simulations.best_model != "gain_loss"]

    simulations_groups = simulations.groupby(["dataset", "index"])
    mappings_groups = mappings.groupby(["dataset", "base_simulation_index", "index"])

    simulations_stats = simulations[["dataset", "index"]].drop_duplicates().set_index(["dataset", "index"])
    mappings_stats = mappings[["dataset", "base_simulation_index", "index"]].set_index(
        ["dataset", "base_simulation_index", "index"]
    )

    print(f"computing frac_terminal_polyploidizations...")
    simulations_frac_terminal_polyploidizations = simulations_groups.parallel_apply(
        get_frac_external_polyploidizations
    ).to_dict()
    simulations_stats["frac_terminal_polyploidizations"] = np.nan
    simulations_stats["frac_terminal_polyploidizations"].fillna(
        value=simulations_frac_terminal_polyploidizations, inplace=True
    )

    mappings_frac_terminal_polyploidizations = mappings_groups.parallel_apply(
        get_frac_external_polyploidizations
    ).to_dict()
    mappings_stats["frac_terminal_polyploidizations"] = np.nan
    mappings_stats["frac_terminal_polyploidizations"].fillna(
        value=mappings_frac_terminal_polyploidizations, inplace=True
    )

    print(f"computing polyploidization_mean_age...")
    simulations_polyploidization_mean_age = simulations_groups.parallel_apply(get_polyploidization_mean_age).to_dict()
    simulations_stats["polyploidization_mean_age"] = np.nan
    simulations_stats["polyploidization_mean_age"].fillna(value=simulations_polyploidization_mean_age, inplace=True)

    mappings_polyploidization_mean_age = mappings_groups.parallel_apply(get_polyploidization_mean_age).to_dict()
    mappings_stats["polyploidization_mean_age"] = np.nan
    mappings_stats["polyploidization_mean_age"].fillna(value=mappings_polyploidization_mean_age, inplace=True)

    print(f"computing mean_terminal_bl...")
    simulations_mean_terminal_bl = simulations_groups.parallel_apply(
        lambda df: get_mean_bl(df, tree_height=tree_height)
    ).to_dict()
    simulations_stats["mean_terminal_bl"] = np.nan
    simulations_stats["mean_terminal_bl"].fillna(value=simulations_mean_terminal_bl, inplace=True)

    mappings_mean_terminal_bl = mappings_groups.parallel_apply(
        lambda df: get_mean_bl(df, tree_height=tree_height)
    ).to_dict()
    mappings_stats["mean_terminal_bl"] = np.nan
    mappings_stats["mean_terminal_bl"].fillna(value=mappings_mean_terminal_bl, inplace=True)

    print(f"computing mean_polyploid_clade_size...")
    simulations_mean_polyploid_clade_size = simulations_groups.parallel_apply(
        lambda rec: get_mean_poly_clade_size(rec, tree, use_relative=False)
    ).to_dict()
    simulations_stats["mean_polyploid_clade_size"] = np.nan
    simulations_stats["mean_polyploid_clade_size"].fillna(value=simulations_mean_polyploid_clade_size, inplace=True)

    mappings_mean_polyploid_clade_size = mappings_groups.parallel_apply(
        lambda rec: get_mean_poly_clade_size(rec, tree, use_relative=False)
    ).to_dict()
    mappings_stats["mean_polyploid_clade_size"] = np.nan
    mappings_stats["mean_polyploid_clade_size"].fillna(value=mappings_mean_polyploid_clade_size, inplace=True)

    simulations_stats.to_csv(processed_simulations_data_path)
    mappings_stats.to_csv(processed_mappings_data_path)


if __name__ == "__main__":
    process_data()
