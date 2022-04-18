import urllib.request
from typing import List
import click
import pandas as pd
import io

def get_data_by_group(group: str) -> pd.DataFrame:
    query_type = "majorGroup"
    response = urllib.request.urlopen(
        "http://ccdb.tau.ac.il/services/countsFull/?" + query_type + "=" + group + "&format=csv")
    res = response.read()
    df = pd.read_csv(io.StringIO(res.decode("utf-8")), dtype={"family":object ,
                                                              "gametophytic":object,
                                                              "genus":object ,
                                                              "id": object,
                                                              "internal_id":object ,
                                                              "major_group":object,
                                                              "matched_name":object,
                                                              "original_name":object ,
                                                              "parsed_n":object,
                                                              "reference":object,
                                                              "resolved_name":object,
                                                              "source":object,
                                                              "sporophytic":object,
                                                              "taxonomic_status":object,
                                                              "voucher":object})
    df["group"] = group
    return df


@click.command()
@click.option(
    "--query_groups",
    help="list of groups to query ccdb data by",
    type=list,
    required=False,
    default=["Gymnosperms", "Pteridophytes", "Bryophytes", "Angiosperms"]
)
@click.option(
    "--output_path",
    help="csv with name resolution results on query names",
    type=click.Path(exists=False),
    required=True,
)
def get_ccdb_data(query_groups: List[str], output_path: str):
    dfs = [get_data_by_group(group=group) for group in query_groups]
    complete_df = pd.concat(dfs)
    complete_df.to_csv(output_path)

if __name__ == "__main__":
    get_ccdb_data()
