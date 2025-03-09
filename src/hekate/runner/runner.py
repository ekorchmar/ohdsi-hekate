from pathlib import Path
import sys
import logging
import datetime

import polars as pl
from runner.cli_args import HekateArgsParser
from csv_read.athena import OMOPVocabulariesV5
from csv_read.source_input import BuildRxEInput
from rx_model.drug_classes import ConceptCodeVocab, ConceptId
from utils.logger import LOGGER, FORMATTER
from utils.constants import VALID_CONCEPT_END_DATE, VALID_CONCEPT_START_DATE

_VERSE = """
With wing of bat and eye of toad,
Avoiding legacy code bloat,
With twig of fern and QA checks,
Hekate starts another hex!
"""


def _int_date_to_str(int_date: int) -> str:
    year = int_date // 10_000
    month = (int_date // 100) % 100
    day = int_date % 100
    return f"{year}-{month:02d}-{day:02d}"


class HekateRunner:
    RESULTS_SCHEMA = {
        "concept_code_1": pl.Utf8,
        "vocabulary_id_1": pl.Utf8,
        "concept_name_1": pl.Utf8,
        "concept_id_2": pl.UInt32,
        "concept_code_2": pl.Utf8,
        "vocabulary_id_2": pl.Utf8,
        "concept_name_2": pl.Utf8,
        # We want to be explicit with DT formatting
        "valid_start_date": pl.Utf8,
        "valid_end_date": pl.Utf8,
        "invalid_reason": pl.Utf8,
    }

    def __init__(self):
        parser = HekateArgsParser()
        self._args = parser.parse_args(sys.argv[1:])
        LOGGER.info(_VERSE)

        self.resulting_mappings: dict[ConceptCodeVocab, list[ConceptId]] = {}

        self.athena_rxne: OMOPVocabulariesV5
        self.build_rxe_source: BuildRxEInput

    def run(self):
        LOGGER.info("Run started")

        self.run_dir: Path
        self._create_run_dir()
        self._attach_file_logging()

        vocab_path = Path(self._args.athena_download_dir)
        source_path = Path(self._args.build_rxe_input_dir)
        self.athena_rxne = OMOPVocabulariesV5(vocab_download_path=vocab_path)
        self.build_rxe_source = BuildRxEInput(
            data_path=source_path,
            athena_vocab=self.athena_rxne,
            delimiter=self._args.delimiter or "\t",
            quote_char=self._args.quote_char or "",
        )

        self.resulting_mappings = self.build_rxe_source.map_to_rxn()

        LOGGER.info("Done")

    def _create_run_dir(self):
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self._args.output_dir / time_now
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def _attach_file_logging(self):
        # If argument -d is passed, enable debug logging to stdout
        if self._args.debug:
            for handler in LOGGER.handlers:
                handler.setLevel(logging.DEBUG)
                LOGGER.debug("Debug logging to stdout enabled")

        log_file = self.run_dir / "hekate_run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(file_handler)
        LOGGER.debug(f"Logging to {log_file}")

    def write_results(self):
        columns = {
            "concept_code_1": [],
            "vocabulary_id_1": [],
            "concept_id_2": [],
        }

        for source, targets in self.resulting_mappings.items():
            columns["concept_code_1"].append(source.concept_code)
            columns["vocabulary_id_1"].append(source.vocabulary_id)
            columns["concept_id_2"].append(
                pl.Series(targets, dtype=self.RESULTS_SCHEMA["concept_id_2"])
            )

        starting_schema = {
            k: v() for k, v in self.RESULTS_SCHEMA.items() if k in columns
        }
        starting_schema |= {"concept_id_2": pl.List(pl.UInt32)}

        mappings_df = pl.LazyFrame(
            columns,
            schema_overrides=starting_schema,
        ).explode("concept_id_2")

        # Attach names and codes from Athena
        mappings_df = mappings_df.join(
            self.athena_rxne.concept.collect().lazy(),
            left_on="concept_id_2",
            right_on="concept_id",
        ).select(
            mappings_df.columns,
            concept_name_2=pl.col("concept_name"),
            concept_code_2=pl.col("concept_code"),
            vocabulary_id_2=pl.col("vocabulary_id"),
        )

        # Attach names from source
        mappings_df = mappings_df.join(
            self.build_rxe_source.dcs.collect().lazy(),
            left_on=["concept_code_1", "vocabulary_id_1"],
            right_on=["concept_code", "vocabulary_id"],
        ).select(
            mappings_df.columns,
            concept_name_1=pl.col("concept_name"),
        )

        # Add static metadata and reorder columns
        mappings_df = mappings_df.with_columns(
            valid_start_date=pl.lit(
                _int_date_to_str(VALID_CONCEPT_START_DATE)
            ).cast(self.RESULTS_SCHEMA["valid_start_date"]),
            valid_end_date=pl.lit(
                _int_date_to_str(VALID_CONCEPT_END_DATE)
            ).cast(self.RESULTS_SCHEMA["valid_start_date"]),
            invalid_reason=pl.lit(None).cast(
                self.RESULTS_SCHEMA["invalid_reason"]
            ),
        )

        print(data := mappings_df.collect())
        data.write_csv(self.run_dir / "hekate_results.csv")
