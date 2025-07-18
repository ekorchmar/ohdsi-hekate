import argparse  # for Typing
import datetime  # for naming directories
import logging  # for logging
import sys  # To read command line arguments
from collections.abc import Sequence, Generator  # for annotations
from pathlib import Path  # for file paths
from itertools import chain  # for flattening results

import polars as pl  # for results writing
import process as p  # for translator and resolver
from csv_read.athena import OMOPVocabulariesV5  # for Athena data
from csv_read.source_input import BuildRxEInput  # for source data
from runner.cli_args import HekateArgsParser  # for command line arguments
from rx_model import drug_classes as dc  # for data classes
from utils.constants import (
    VALID_CONCEPT_END_DATE,  # For valid relations formatting
    VALID_CONCEPT_START_DATE,  # ditto
    CUSTOM_CONCEPT_ID_START,  # To generate (temporary) ids
)
from utils.exceptions import (
    ForeignPackCreationError,
    ResolutionError,
    UnmappedSourceConceptError,
)
from utils.logger import FORMATTER, LOGGER
from utils.utils import int_date_to_str

_VERSE = """
With wing of bat and eye of toad,
Avoiding legacy code bloat,
With twig of fern and QA checks,
Hekate starts another hex!
"""

type _InterimResult[T, U] = dict[
    T, dict[U, Sequence[dc.HierarchyNode[dc.ConceptId]]]
]

type _InterimDrugResult = _InterimResult[
    dc.ForeignNodePrototype, dc.ForeignDrugNode[dc.Strength | None]
]

type _InterimPackResult = _InterimResult[
    dc.ForeignPackNodePrototype, dc.ForeignPackNode
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

        self.resulting_drug_mappings: dict[
            dc.ConceptCodeVocab, list[dc.HierarchyNode[dc.ConceptId]]
        ] = {}
        self.resulting_pack_mappings: dict[
            dc.ConceptCodeVocab, list[dc.HierarchyNode[dc.ConceptId]]
        ] = {}

        self.__id_count = CUSTOM_CONCEPT_ID_START

        self.athena_rxne: OMOPVocabulariesV5
        self.build_rxe_source: BuildRxEInput

        self.concepts_to_graph: list[dc.ConceptCodeVocab] = []

    def new_concept_id(self) -> Generator[dc.ConceptId]:
        while True:
            yield dc.ConceptId(self.__id_count)
            self.__id_count += 1

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

        drug_node_options = self._find_drug_node_mappings()

        self.resulting_drug_mappings = self._resolve_nodes(
            drug_node_options,
            p.DrugResolver,
        )
        # Reshape the options to provide translation from source code & vocab
        drug_translations = {
            drug_prototype.identifier: translations
            for drug_prototype, translations in drug_node_options.items()
        }
        pack_node_options = self._find_pack_options(drug_translations)

        self.resulting_pack_mappings = self._resolve_nodes(
            pack_node_options,
            p.PackResolver,
        )

        self.write_results()
        self.write_report()

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
        columns: dict[str, list[str | pl.Series]] = {
            "concept_code_1": [],
            "vocabulary_id_1": [],
            "concept_id_2": [],
        }

        for source, targets in chain(
            self.resulting_drug_mappings.items(),
            self.resulting_pack_mappings.items(),
        ):
            columns["concept_code_1"].append(source.concept_code)
            columns["vocabulary_id_1"].append(source.vocabulary_id)
            columns["concept_id_2"].append(
                pl.Series(
                    [target.identifier for target in targets],
                    dtype=self.RESULTS_SCHEMA["concept_id_2"],
                )
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

    def write_report(self):
        """
        Write simple stats describing the source concepts and their resulting mapping

        Columns:
            - Source concept class: coalesce of drug_concept_stage source_concept_class and concept_class columns
            - Target concept class: concept_class_id of the resulting mapping
            - Target Vocab: vocabulary_id of the resulting mapping
            - Count: number of mappings for each source concept class
        """
        # Collect all mappings from both drug and pack results
        all_mappings: list[dict[str, int | str | None]] = []

        for source_concept, target_nodes in chain(
            self.resulting_drug_mappings.items(),
            self.resulting_pack_mappings.items(),
        ):
            for target_node in target_nodes:
                all_mappings.append({
                    "source_concept_code": source_concept.concept_code,
                    "source_vocabulary_id": source_concept.vocabulary_id,
                    "target_concept_id": target_node.identifier,
                })

        # Get all source concepts from build_rxe_source
        all_source_concepts = self.build_rxe_source.dcs.collect().select(
            "concept_code",
            "vocabulary_id",
            "concept_class_id",
            "source_concept_class_id",
        )

        # Find unmapped source concepts
        if all_mappings:
            mapped_concepts = (
                pl.DataFrame(all_mappings)
                .select("source_concept_code", "source_vocabulary_id")
                .unique()
            )

            unmapped_concepts = all_source_concepts.join(
                mapped_concepts,
                left_on=["concept_code", "vocabulary_id"],
                right_on=["source_concept_code", "source_vocabulary_id"],
                how="anti",
            )

            # Add unmapped concepts to all_mappings with special markers
            for row in unmapped_concepts.iter_rows(named=True):
                all_mappings.append({
                    "source_concept_code": row["concept_code"],
                    "source_vocabulary_id": row["vocabulary_id"],
                    "target_concept_id": None,  # Will be handled as unmapped
                })

        if not all_mappings:
            self.logger.warning("No mappings found for report generation")
            return

        # Create DataFrame from mappings
        mappings_df = pl.DataFrame(all_mappings)

        # Join with target concept data to get target concept classes
        target_concepts = self.athena_rxne.concept.collect().select(
            pl.col("concept_id").alias("target_concept_id"),
            pl.col("concept_class_id").alias("target_concept_class_id"),
            pl.col("vocabulary_id").alias("target_vocabulary_id"),
        )

        # Create the report by joining all data
        report_df = (
            mappings_df.join(
                all_source_concepts,
                left_on=["source_concept_code", "source_vocabulary_id"],
                right_on=["concept_code", "vocabulary_id"],
                how="left",
            )
            .join(
                target_concepts,
                left_on="target_concept_id",
                right_on="target_concept_id",
                how="left",
            )
            .with_columns(
                source_concept_class=pl.coalesce(
                    pl.col("source_concept_class_id"),
                    pl.col("concept_class_id"),
                ),
                target_concept_class=pl.coalesce(
                    pl.col("target_concept_class_id"),
                    pl.lit("UNMAPPED"),
                ),
                target_vocabulary=pl.coalesce(
                    pl.col("target_vocabulary_id"),
                    pl.lit("UNMAPPED"),
                ),
            )
            .group_by(
                "source_concept_class",
                "target_concept_class",
                "target_vocabulary",
            )
            .len("count")
            .sort(
                "source_concept_class",
                "target_concept_class",
                "target_vocabulary",
            )
            .select(
                source_concept_class=pl.col("source_concept_class"),
                target_concept_class=pl.col("target_concept_class"),
                target_vocabulary=pl.col("target_vocabulary"),
                count=pl.col("count"),
            )
        )

        # Write the report
        report_path = self.run_dir / "hekate_mapping_report.csv"
        report_df.write_csv(report_path)

        # Log summary statistics
        total_concepts = report_df.select(pl.sum("count")).item()
        total_mapped = (
            report_df.filter(pl.col("target_concept_class") != "UNMAPPED")
            .select(pl.sum("count"))
            .item()
        )
        total_unmapped = (
            report_df.filter(pl.col("target_concept_class") == "UNMAPPED")
            .select(pl.sum("count"))
            .item()
        )
        unique_source_classes = report_df.select(
            pl.n_unique("source_concept_class")
        ).item()

        self.logger.info(f"Mapping report written to {report_path}")
        self.logger.info(f"""Brief:
 - Total source concepts: {total_concepts:,}")
 - Mapped concepts: {total_mapped:,}
 - Unmapped concepts: {total_unmapped:,}
 - Unique source concept classes: {unique_source_classes}
 """)
        print(report_df)

    def _find_drug_node_mappings(self) -> _InterimDrugResult:
        """
        Map the generated nodes to all possible RxNorm concepts.

        Generates non-disambiguated mappings to all terminals, returning them as
        a 2 level nested dictionary, where first level is indexed by the
        source-native ForeignNodePrototype representation, second level is the
        variation of translated ForeignNode, and final value is a list of
        DrugNodes of different class.
        """

        result: _InterimDrugResult = {}

        for prototype in self.build_rxe_source.prepare_drug_nodes(
            crash_on_error=False
        ):
            result[prototype] = {}
            translated_nodes = self.translator.translate_drug_node(
                prototype, self.new_concept_id()
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
                                f"{prototype.identifier.concept_code}.svg"
                            ),
                            use_identifier=prototype.identifier,
                        )

                result[prototype][option] = list(node_result.values())

        return result

    def _find_pack_options(
        self,
        drug_node_results: dict[
            dc.ConceptCodeVocab,
            dict[
                dc.ForeignDrugNode[dc.Strength | None],
                Sequence[dc.HierarchyNode[dc.ConceptId]],
            ],
        ],
    ) -> _InterimPackResult:
        """
        Map the generated nodes to all possible RxNorm concepts.

        Generates non-disambiguated mappings to all terminals, returning them as
        a 2 level nested dictionary, where first level is indexed by the
        source-native ForeignNodePrototype representation, second level is the
        variation of translated ForeignPackNode, and final value is a list of
        PackNodes of different class.
        """

        result: _InterimPackResult = {}

        for prototype in self.build_rxe_source.prepare_pack_nodes(
            crash_on_error=False
        ):
            result[prototype] = {}
            translated_nodes = self.translator.translate_pack_node(
                prototype, drug_node_results, self.new_concept_id()
            )

            while True:
                try:
                    option = next(translated_nodes)
                except StopIteration:
                    break

                except ForeignPackCreationError as e:
                    self.logger.error(
                        f"Pack Node {prototype.identifier} could not be mapped "
                        f"to RxNorm hierarchy: {e}"
                    )
                    continue  # to the next prototype

                except UnmappedSourceConceptError as e:
                    # This is expected for now
                    # TODO: skip all permutations of the node
                    self.logger.error(
                        f"Node {prototype.identifier} could not be mapped to "
                        f"RxNorm: {e}"
                    )
                    continue  # to the next prototype

                visitor = p.PackNodeFinder(
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
                                f"{prototype.identifier.concept_code}.svg"
                            ),
                            use_identifier=prototype.identifier,
                        )

                result[prototype][option] = list(node_result.values())

        return result

    def _resolve_nodes(
        self,
        terminals: _InterimDrugResult | _InterimPackResult,
        resolver_class: type[p.DrugResolver | p.PackResolver],
    ) -> dict[dc.ConceptCodeVocab, list[dc.HierarchyNode[dc.ConceptId]]]:
        """
        Disambiguate the terminal drug node options and return the final
        mappings.
        """
        out: dict[
            dc.ConceptCodeVocab, list[dc.HierarchyNode[dc.ConceptId]]
        ] = {}
        for prototype, node_mappings in terminals.items():
            if not node_mappings:
                self.logger.error(
                    f"No mappings found for {prototype.identifier}"
                )
                continue

            resolver = resolver_class(
                source_prototype=prototype,  # pyright: ignore[reportArgumentType]  # noqa: E501
                mapping_results=node_mappings,  # pyright: ignore[reportArgumentType]  # noqa: E501
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
                    node for node in resulting_mappings
                ]
        return out
