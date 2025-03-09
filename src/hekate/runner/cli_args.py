import argparse
import pathlib

_DESCRIPTION = """\
A software tool to map drug content and build RxNorm/RxNorm Extension
hierarchies in OMOP CDM context.
"""


def _get_version() -> str:
    """Get the version of the package."""
    # This is a bit brittle, but doing anything more sophisticated would require
    # a non-standard build tool.
    with open(pathlib.Path(__file__).parent.parent / "__version__") as f:
        return f.read().strip()


class HekateArgsParser(argparse.ArgumentParser):
    """Parser for command line arguments for main entry point of Hekate CLI."""

    def __init__(self):
        super().__init__(
            prog="Hekate " + _get_version(),
            description=_DESCRIPTION,
            formatter_class=argparse.RawTextHelpFormatter,
            usage=(
                "hekate -a <athena-download-dir> -b <build-rxe-input-dir> "
                " [options]"
            ),
        )

        # Required arguments -- input and output paths
        self.add_argument(
            "-a",
            "--athena-download-dir",
            type=pathlib.Path,
            help=(
                "Path to the directory where Athena (https://athena.ohdsi.org) "
                "downloads are stored. CPT4 reconstitution is not required. "
                "RxNorm and UCUM vocabularies must be included in the selected "
                "Subset, and RxNorm Extension is also recommended if you are "
                "using non-US drug data or use Hekate for vocabulary authoring."
            ),
            dest="athena_download_dir",
            required=True,
        )
        self.add_argument(
            "-b",
            "--build-rxe-input-dir",
            type=pathlib.Path,
            help=(
                "Path to where the CSV/TSV files containing the source data "
                "in format described at International Drug Vocabulary "
                "Implementation Process ("
                "https://github.com/OHDSI/Vocabulary-v5.0/wiki/International-Drug-Vocabulary-Implementation-Process"
                ") are stored."
            ),
            dest="build_rxe_input_dir",
            required=True,
        )
        self.add_argument(
            "-o",
            "--output-dir",
            type=pathlib.Path,
            help=(
                "Path to the directory where the output files will be stored. "
                "The output files are currently limited to the debug trace "
                "run log and a single conept_relationship_stage.csv file, "
                "which contains reviewable superset of columns of the "
                "corresponding GenericUpdate.sql input table. Default is "
                "run/ subdirectory of the current working directory."
            ),
            dest="output_dir",
            default=pathlib.Path.cwd() / "run",
        )

        # Optional arguments
        self.add_argument(
            "-d",
            "--debug",
            action="store_true",
            help=(
                "Enable debug logging to stdout. The log file will always "
                "contain debug trace."
            ),
            default=False,
            dest="debug",
        )
        self.add_argument(
            "-s--separator",
            type=str,
            help=(
                "Separator character used in the BuildRxE input files. Default "
                "is tab matching Athena download format."
            ),
            default="\t",
            dest="delimiter",
        )
        self.add_argument(
            "-q",
            "--quotechar",
            type=str,
            help=(
                "Quote character used in the BuildRxE input files. Default is "
                "explicit lack of quoting matching Athena download format."
            ),
            default=None,
            dest="quote_char",
        )
