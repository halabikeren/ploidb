import os
from typing import Optional

import sqlite3

import numpy as np
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=5, use_memory_fs=False)


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from Bio import Entrez

import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv(f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/.env")


def get_itis_db(db_link: str, db_dir: str):
    if not os.path.exists(f"{db_dir}/itisSqlite.zip"):
        res = os.system(f"cd {db_dir}; wget {db_link}; unzip itisSqlite.zip")
    db_prefix = [
        path
        for path in os.listdir(db_dir)
        if path.startswith("itis") and os.path.isdir(f"{db_dir}/{path}")
    ][0]
    db_path = f"{db_dir}/{db_prefix}/ITIS.sqlite"
    return db_path


def get_plant_kingdom_id(db_connection: sqlite3.Connection) -> int:
    return pd.read_sql_query(
        "SELECT kingdom_id from kingdoms WHERE kingdom_name LIKE 'Plant%' --case-insensitive;",
        db_connection,
    ).values[0][0]


def get_rank_data(db_connection: sqlite3.Connection, kingdom_id: int) -> pd.DataFrame:
    rank_data_query = f"SELECT rank_id, rank_name FROM taxon_unit_types WHERE kingdom_id IS {kingdom_id};"
    ranks_data = pd.read_sql_query(rank_data_query, db_connection)
    ranks_data.rank_name = ranks_data.rank_name.str.lower()
    return ranks_data


def get_taxonomic_data(
    db_connection: sqlite3.Connection, kingdom_id: int, names: list[str]
):
    taxonomic_units_quey = (
        f"SELECT * FROM taxonomic_units WHERE kingdom_id IS {kingdom_id}"
    )
    taxonomic_units_data = pd.read_sql_query(taxonomic_units_quey, db_connection)
    taxonomic_units_data.complete_name = (
        taxonomic_units_data.complete_name.str.capitalize()
    )
    relevant_taxonomic_data = taxonomic_units_data.loc[
        taxonomic_units_data.complete_name.isin(names)
    ]
    if len(names) > 0:
        logger.info(
            f"% names covered by db taxonomic data = {np.round(relevant_taxonomic_data.shape[0] / len(names) * 100, 2)}%"
        )
    return taxonomic_units_data


def get_rank_name(
    item: pd.Series,
    rank_name: str,
    ranks_data: pd.DataFrame,
    taxonomic_data: pd.DataFrame,
) -> str:
    ordered_ranks = ranks_data.sort_values("rank_id")["rank_name"].to_list()
    tax_order_diff = ordered_ranks.index(rank_name) - ordered_ranks.index(
        item.rank_name
    )
    if (
        tax_order_diff < 0
    ):  # the rank of item is higher than that of the required rank name, so it cannot be a subset of rank_name
        return np.nan
    while tax_order_diff > 0:  # go up by one rank
        matches = taxonomic_data.loc[taxonomic_data.tsn == item.parent_tsn]
        if matches.shape[0] == 0:
            return np.nan
        item = matches.iloc[0]
        tax_order_diff = ordered_ranks.index(rank_name) - ordered_ranks.index(
            item.rank_name
        )  # re-compute the taxonomic diff
    if tax_order_diff < 0:
        return np.nan
    return item.complete_name


def get_ncbi_tax_id(tax_name: str) -> int:
    Entrez.email = os.getenv("ENTREZ_EMAIL")
    tax_id = np.nan
    try:
        res = Entrez.read(
            Entrez.esearch(
                db="taxonomy",
                term=tax_name,
                retmode="xml",
                api_key=os.environ.get("NCBI_API_KEY"),
            )
        )["IdList"]
        if len(res) > 0:
            tax_id = int(res[0])
    except Exception as e:
        logger.warning(
            f"could not retrieve ncbi tax id fro {tax_name} due to error {e}"
        )
    return tax_id


def fill_missing_data_from_itis(
    input_df: pd.DataFrame, input_col: str, db_link: str, db_dir: str
):

    db_path = get_itis_db(db_link=db_link, db_dir=db_dir)
    connection = sqlite3.connect(db_path)

    plant_kingdom_id = get_plant_kingdom_id(db_connection=connection)

    ranks_data = get_rank_data(db_connection=connection, kingdom_id=plant_kingdom_id)
    rank_id_to_name = ranks_data.set_index("rank_id")["rank_name"].to_dict()

    input_df[f"{input_col}_capitalized"] = input_df[input_col].str.capitalize()
    taxonomic_data = get_taxonomic_data(
        db_connection=connection,
        kingdom_id=plant_kingdom_id,
        names=input_df[f"{input_col}_capitalized"].to_list(),
    )
    taxonomic_data["rank_name"] = taxonomic_data["rank_id"].apply(
        lambda rank_id: rank_id_to_name[rank_id]
    )
    taxonomic_data["genus_name"] = taxonomic_data.apply(
        lambda row: get_rank_name(
            item=row,
            rank_name="genus",
            ranks_data=ranks_data,
            taxonomic_data=taxonomic_data,
        ),
        axis=1,
    )
    taxonomic_data["family_name"] = taxonomic_data.apply(
        lambda row: get_rank_name(
            item=row,
            rank_name="family",
            ranks_data=ranks_data,
            taxonomic_data=taxonomic_data,
        ),
        axis=1,
    )

    input_df.set_index(f"{input_col}_capitalized", inplace=True)
    for col in ["taxon_rank", "genus", "family"]:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df["taxon_rank"].fillna(
        value=taxonomic_data.set_index("complete_name")["rank_name"].to_dict(),
        inplace=True,
    )
    input_df["genus"].fillna(
        value=taxonomic_data.set_index("complete_name")["genus_name"].to_dict(),
        inplace=True,
    )
    input_df["family"].fillna(
        value=taxonomic_data.set_index("complete_name")["family_name"].to_dict(),
        inplace=True,
    )
    return input_df


def get_tax_id_to_tax_data(tax_ids: list[int]) -> tuple[dict[int, str],
                                                        dict[int, str],
                                                        dict[int, str]]:
    tax_id_to_genus, tax_id_to_family, tax_id_to_rank = dict(), dict(), dict()
    Entrez.email = os.getenv("ENTREZ_EMAIL")
    try:
        res = list(
            Entrez.parse(
                Entrez.efetch(
                    db="taxonomy",
                    id=",".join([str(i) for i in tax_ids]),
                    retmode="xml",
                    api_key=os.environ.get("NCBI_API_KEY"),
                )
            )
        )
        for record in res:
            tax_id = int(record["TaxId"])
            tax_id_to_rank[tax_id] = record["Rank"].lower()
            for rank_data in record["LineageEx"]:
                if rank_data["Rank"] == "genus":
                    tax_id_to_genus[tax_id] = rank_data["ScientificName"].lower()
                if rank_data["Rank"] == "family":
                    tax_id_to_family[tax_id] = rank_data["ScientificName"].lower()
    except Exception as e:
        logger.warning(
            f"could not fetch tax data for {len(tax_ids)} tax ids due to error {e}"
        )
    return tax_id_to_genus, tax_id_to_family, tax_id_to_rank

def fill_missing_data_from_ncbi(data: pd.DataFrame, search_by_col: str) -> pd.DataFrame:
    data.reset_index(inplace=True)
    data["ncbi_tax_id"] = data[search_by_col].parallel_apply(get_ncbi_tax_id)
    tax_ids = data.ncbi_tax_id.dropna().astype(np.int16).unique().tolist()
    batch_size = 1000
    tax_ids_batches = [
        tax_ids[i : i + batch_size] for i in range(0, len(tax_ids), batch_size)
    ]
    tax_id_to_genus, tax_id_to_family, tax_id_to_rank = dict(), dict(), dict()
    for batch in tax_ids_batches:
        batch_to_genus, batch_to_family, batch_to_rank = get_tax_id_to_tax_data(tax_ids=batch)
        tax_id_to_genus.update(batch_to_genus)
        tax_id_to_family.update(batch_to_family)
        tax_id_to_rank.update(batch_to_rank)
    data.set_index("ncbi_tax_id", inplace=True)
    data["genus"].fillna(value=tax_id_to_genus, inplace=True)
    data["family"].fillna(value=tax_id_to_family, inplace=True)
    if "taxon_rank" in data.columns:
        data["taxon_rank"].fillna(value=tax_id_to_rank, inplace=True)
    data.genus = data.parallel_apply(lambda record: record[search_by_col] if record.taxon_rank == "genus" else np.nan, axis=1)
    data.family = data.parallel_apply(lambda record: record[search_by_col] if record.taxon_rank == "family" else np.nan, axis=1)
    return data


def add_taxonomic_data(
    input_df: pd.DataFrame, input_col: str, itis_db_dir: Optional[str]
) -> pd.DataFrame:
    if itis_db_dir is None:
        itis_db_dir = os.path.dirname(os.getcwd())
    fill_missing_data_from_itis(
        input_df=input_df,
        input_col=input_col,
        db_link=os.environ.get("ITIS_DB_LINK"),
        db_dir=itis_db_dir,
    )
    missing_data = input_df.loc[(input_df.genus.isna()) | (input_df.family.isna())]
    complementary_data = fill_missing_data_from_ncbi(
        data=missing_data, search_by_col=input_col
    )
    complementary_data.to_csv(f"{os.getcwd()}/complementary_tax_data.csv")

    complementary_data.set_index(input_df.index.name, inplace=True)
    input_df.update(complementary_data)
    input_df.reset_index(inplace=True)
    input_df.drop(f"{input_col}_capitalized", axis=1, inplace=True)
    return input_df
