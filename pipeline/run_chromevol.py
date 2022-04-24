import json
import click
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_generation.chromevol import ChromevolExecutor

@click.command()
@click.option(
    "--input_path",
    help="csv with query names to apply name resolution on",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
def run_chromevol(input_path: str):
    with open(input_path, "r") as infile:
        input_args = json.load(fp=infile)
    ChromevolExecutor.run(input_args=input_args)

if __name__ == '__main__':
    run_chromevol()

