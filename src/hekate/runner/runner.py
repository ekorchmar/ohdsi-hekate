import argparse  # for Typing
import datetime  # for naming directories
import logging  # for logging
import sys  # To read command line arguments
from collections.abc import Sequence  # for Typing
from pathlib import Path  # for file paths

import polars as pl  # for resultss writing
import process as p  # for translator and resolver
from csv_read.athena import OMOPVocabulariesV5  # for Athena data
from csv_read.source_input import BuildRxEInput  # for source data
from runner.cli_args import HekateArgsParser  # for command line arguments
from rx_model import drug_classes as dc  # for data classes
from utils.constants import VALID_CONCEPT_END_DATE, VALID_CONCEPT_START_DATE
from utils.exceptions import UnmappedSourceConceptError, ResolutionError
from utils.logger import FORMATTER, LOGGER
from utils.utils import int_date_to_str

_VERSE = """
With wing of bat and eye of toad,
Avoiding legacy code bloat,
With twig of fern and QA checks,
Hekate starts another hex!
"""

type _InterimResult = dict[
    dc.ForeignNodePrototype,
    dict[
        dc.ForeignDrugNode[dc.Strength | None],
        Sequence[dc.DrugNode[dc.ConceptId, dc.Strength | None]],
    ],
]


class HekateRunner:
    RESULTS_SCHEMA: dict[str, type[pl.DataType]] = {
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
        self._args: argparse.Namespace = parser.parse_args(sys.argv[1:])
        LOGGER.info(_VERSE)

        self.resulting_mappings: dict[
            dc.ConceptCodeVocab, list[dc.ConceptId]
        ] = {}

        self.athena_rxne: OMOPVocabulariesV5
        self.build_rxe_source: BuildRxEInput

        self.concepts_to_graph: list[dc.ConceptCodeVocab] = []

    def run(self):
        self.logger: logging.Logger = LOGGER.getChild(self.__class__.__name__)
        self.logger.info("Run started")

        # Output graphs if requested
        if gsc := self._args.graph_source_concepts:
            assert isinstance(gsc, str)
            self.concepts_to_graph.extend(
                map(
                    lambda v_c: dc.ConceptCodeVocab(*v_c.split(":")[::-1]),
                    gsc.split(";"),
                )
            )

        self.run_dir: Path
        self._create_run_dir()
        self._attach_file_logging()

        vocab_path = Path(self._args.athena_download_dir)
        source_path = Path(self._args.build_rxe_input_dir)
        self.athena_rxne = OMOPVocabulariesV5(vocab_download_path=vocab_path)
        self.build_rxe_source = BuildRxEInput(
            data_path=source_path,
            delimiter=self._args.delimiter or "\t",
            quote_char=self._args.quote_char or "",
        )

        self.translator: p.NodeTranslator = p.NodeTranslator(
            rx_atoms=self.athena_rxne.atoms, logger=self.logger
        )
        self.translator.read_translations(source=self.build_rxe_source)

        terminals = self._find_terminals()
        self.resulting_mappings = self._resolve(terminals)

        LOGGER.info("Done")

    def _create_run_dir(self):
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self._args.output_dir / time_now
        self.run_dir.mkdir(parents=True, exist_ok=False)

        if self.concepts_to_graph:
            (self.run_dir / "graphs").mkdir()

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
                int_date_to_str(VALID_CONCEPT_START_DATE)
            ).cast(self.RESULTS_SCHEMA["valid_start_date"]),
            valid_end_date=pl.lit(int_date_to_str(VALID_CONCEPT_END_DATE)).cast(
                self.RESULTS_SCHEMA["valid_start_date"]
            ),
            invalid_reason=pl.lit(None).cast(
                self.RESULTS_SCHEMA["invalid_reason"]
            ),
        ).select(list(self.RESULTS_SCHEMA.keys()))

        print(data := mappings_df.collect())
        data.write_csv(self.run_dir / "hekate_results.csv")

    def _find_terminals(self) -> _InterimResult:
        """
        Map the generated nodes to RxNorm concepts.
        """

        result: _InterimResult = {}

        # 2 billion is conventionally used for local concept IDs
        def new_concept_id():
            two_bill = 2_000_000_000
            while True:
                yield dc.ConceptId(two_bill)
                two_bill += 1

        cid_counter = new_concept_id()
        for prototype in self.build_rxe_source.prepare_drug_nodes(
            crash_on_error=False
        ):
            result[prototype] = {}
            translated_nodes = self.translator.translate_node(
                prototype, lambda: next(cid_counter)
            )
            while True:
                try:
                    option = next(translated_nodes)
                except StopIteration:
                    break
                except UnmappedSourceConceptError as e:
                    # This is expected for now
                    # TODO: skip all permutations of the node
                    self.logger.error(
                        f"Node {prototype.identifier} could not be mapped to "
                        f"RxNorm: {e}"
                    )
                    continue

                visitor = p.DrugNodeFinder(
                    option,
                    self.athena_rxne.hierarchy,
                    self.logger,
                    save_subplot=prototype.identifier in self.concepts_to_graph,
                )
                visitor.start_search()

                node_result = visitor.get_search_results()

                # If the node has no results yet and is supposed to be graphed,
                # do it now -- we only graph the first option
                if prototype.identifier in self.concepts_to_graph:
                    if not result[prototype]:
                        visitor.draw_subgraph(
                            self.run_dir
                            / "graphs"
                            / (
                                f"{prototype.identifier.vocabulary_id}_"
                                + f"{prototype.identifier.concept_code}.svg"
                            ),
                            use_identifier=prototype.identifier,
                        )

                result[prototype][option] = list(node_result.values())

        return result

    def _resolve(
        self, terminals: _InterimResult
    ) -> dict[dc.ConceptCodeVocab, list[dc.ConceptId]]:
        """
        Disambiguate the terminal nodes and return the final mappings.
        """
        out: dict[dc.ConceptCodeVocab, list[dc.ConceptId]] = {}
        for prototype, node_mappings in terminals.items():
            if not node_mappings:
                self.logger.error(
                    f"No mappings found for {prototype.identifier}"
                )
                continue

            resolver = p.Resolver(
                source_definition=prototype,
                mapping_results=node_mappings,
                logger=self.logger,
                concept_handle=self.athena_rxne.concept,
            )

            try:
                resulting_mappings = resolver.pick_omop_mapping()
            except ResolutionError as e:
                self.logger.error(
                    f"Could not resolve mappings for {prototype.identifier}: "
                    f"{e}"
                )
                continue
            else:
                self.logger.debug(
                    f"Resolved {len(resulting_mappings)} mappings for "
                    f"{prototype.identifier}"
                )
                out[prototype.identifier] = [
                    node.identifier for node in resulting_mappings
                ]
        return out
