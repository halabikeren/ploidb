import json
import click
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_generation.one_two_tree import OneTwoTreeExecutor


@click.command()
@click.option(
    "--input_path",
    help="json path with input for one two tree",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
def run_onw_two_tree(input_path: str):
    with open(input_path, "r") as infile:
        input_args = json.load(fp=infile)
    OneTwoTreeExecutor.run(input_args=input_args)


if __name__ == "__main__":
    run_onw_two_tree()
