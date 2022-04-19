import json
import click
from data_generation.chromevol import ChromevolExecutor

click.command()
@click.option(
    "--input_path",
    help="csv with query names to apply name resolution on",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=True,
)
def run_chromevol(input_path: str):
    with open(input_path, "r") as infile:
        input_args = json.load(file=infile)
    ChromevolExecutor.run(input_args=input_args)

if __name__ == '__main__':
    run_chromevol()

