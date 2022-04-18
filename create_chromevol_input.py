import logging
logger = logging.getLogger(__name__)

import sys
import os
import click
from typing import Dict, List

import pandas as pd
import numpy as np
from ete3 import Tree

from collections import Counter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



def read_tree(tree_path: str, unresolved_to_resolved_names_translator: Dict[str, str]) -> Tree:
    tree = Tree(tree_path, format=1)
    orig_num_tips = len(tree.get_leaf_names())
    logger.info(f"# original unresolved unique names = {orig_num_tips:,}")
    for leaf in tree.get_leaves():
        leaf.name = leaf.name.replace("_", " ").lower()
        if leaf.name in unresolved_to_resolved_names_translator:
            leaf.name = unresolved_to_resolved_names_translator[leaf.name]
    new_num_tips = len(set(tree.get_leaf_names()))
    logger.info(f"# resolved unique names = {new_num_tips:,}")
    logger.info(f"fraction of unresolved names that are mapped to a single unique name = {np.round(len(set(tree.get_leaf_names()))/orig_num_tips*100,2)}% ({len(set(tree.get_leaf_names())):,}/{orig_num_tips:,})")
    return tree

def process_ccdb(ccdb_path: str, resolved_names: pd.DataFrame) -> pd.DataFrame:
    ccdb = pd.read_csv(ccdb_path)
    ccdb["parsed_n"] = pd.to_numeric(ccdb["parsed_n"], errors="coerce")
    logger.info(f"# ccdb records = {ccdb.shape[0]:,}")
    logger.info(f"ccdb unique resolved names by taxonome = {len(ccdb['resolved_name'].unique()):,}")
    ccdb["current_resolved_name"] = np.nan
    ccdb["current_genus"] = np.nan
    ccdb["current_family"] = np.nan
    ccdb["original_name"] = ccdb["original_name"].str.lower()
    ccdb.set_index("original_name", inplace=True)
    ccdb["current_resolved_name"].fillna(value=resolved_names.set_index("query")["matched_name"].to_dict(), inplace=True)
    ccdb["current_genus"].fillna(value=resolved_names.set_index("query")["genus"].to_dict(), inplace=True)
    ccdb["current_family"].fillna(value=resolved_names.set_index("query")["family"].to_dict(), inplace=True)
    ccdb.reset_index(inplace=True)
    logger.info(f"failed to resolve {ccdb.loc[ccdb.current_resolved_name.isna()].shape[0]:,} names")
    logger.info(f"n# ccdb unique resolved names currently = {len(ccdb['current_resolved_name'].unique()):,}")
    return ccdb

def get_distribution_str(chromosome_numbers: List[int]):
    counter = Counter(chromosome_numbers)
    total_nums = len(chromosome_numbers)
    dist_str = ""
    for num in counter.keys():
        dist_str += f"{num}={np.round(counter[num]/total_nums,2)}_"
    return dist_str[:-1]

def write_chromevol_input_by_taxonomic_rank(ccdb_data: pd.DataFrame, tree: Tree, taxonomic_rank_to_group_by: str, write_dir: str):
    relevant_ccdb_data_by_rank = ccdb_data.groupby(taxonomic_rank_to_group_by)
    chromevol_input_dir = f"{write_dir}/by_{taxonomic_rank_to_group_by}/"
    os.makedirs(chromevol_input_dir, exist_ok=True)
    for group in relevant_ccdb_data_by_rank.groups.keys():
        genus_dir = f"{chromevol_input_dir}/{group}/"
        os.makedirs(genus_dir, exist_ok=True)
        ccdb_group_data = relevant_ccdb_data_by_rank.get_group(group)
        names = [name.rstrip() for name in ccdb_group_data.current_resolved_name.unique()]
        group_tree = tree.copy()
        try:
            group_tree.prune(names)
            counts_med_path, count_med = f"{genus_dir}/counts_median.fasta", []
            counts_dist_path, counts_dist = f"{genus_dir}/counts_distribution.fasta", []
            tree_path = f"{genus_dir}/tree.nwk"
            for leaf in group_tree.get_leaves():
                chromosome_numbers = ccdb_group_data.loc[
                    ccdb_group_data.current_resolved_name == leaf.name, 'parsed_n'].dropna().values.tolist()
                leaf.add_features(pr_name="chromosome_number", pr_value=chromosome_numbers)
                count_med.append(SeqRecord(id=leaf.name, name=leaf.name, description=leaf.name,
                                           seq=Seq(str(np.median(chromosome_numbers)))))
                counts_dist.append(SeqRecord(id=leaf.name, name=leaf.name, description=leaf.name,
                                             seq=Seq(get_distribution_str(chromosome_numbers=chromosome_numbers))))
            group_tree.write(outfile=tree_path)
            SeqIO.write(count_med, counts_med_path, format="fasta")
            SeqIO.write(counts_dist, counts_dist_path, format="fasta")
        except Exception as e:
            logger.error(f"failed to parse output for {group} due to error {e} and will thus skip it")
            continue


@click.command()
@click.option(
    "--taxonomic_rank_to_group_by",
    help="taxonomic rank to group ccdb records by",
    type=click.Choice(["genus", "family"]),
    required=False,
    default="genus"
)
@click.option(
    "--tree_path",
    help="path to the tree that should be used as input to prune subtrees from",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--resolved_names_path",
    help="path to csv file with the resolved names",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--ccdb_path",
    help="path to ccdb data",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
@click.option(
    "--output_dir",
    help="directory to create the chromevol input in",
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    "--log_path",
    help="path to log file of the script",
    type=click.Path(exists=False),
    required=True,
)
def create_chromevol_input(taxonomic_rank_to_group_by: str, tree_path: str, resolved_names_path: str, ccdb_path: str, output_dir: str, log_path: str):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(str(log_path)), ],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    resolved_names = pd.read_csv(resolved_names_path)
    logger.info(f"processed {resolved_names.shape[0]:,} resolved names")

    ccdb = process_ccdb(ccdb_path=ccdb_path, resolved_names=resolved_names)
    ccdb_names = set(ccdb.current_resolved_name.unique())

    tree = read_tree(tree_path=tree_path, unresolved_to_resolved_names_translator=resolved_names.set_index("query")["matched_name"].to_dict())
    tree_names = set(tree.get_leaf_names())

    ccdb_names_in_tree = ccdb_names.intersection(tree_names)
    logger.info(
        f"% ccdb records that are included in the tree = {np.round(len(ccdb_names_in_tree) / len(ccdb_names) * 100, 2)}% ({len(ccdb_names_in_tree)}/{len(ccdb_names)})")
    relevant_ccdb_data = ccdb.loc[ccdb.current_resolved_name.isin(list(ccdb_names_in_tree))]

    write_chromevol_input_by_taxonomic_rank(ccdb_data=relevant_ccdb_data, tree=tree, taxonomic_rank_to_group_by=taxonomic_rank_to_group_by, write_dir=output_dir)

if __name__ == '__main__':
    create_chromevol_input()