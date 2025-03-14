"""
Contains implementations to read TSV data from a file and transform it
into ForeignDrugNode objects for evaluation.
"""

import logging
from abc import ABC
from collections.abc import Generator
from pathlib import Path
from typing import Annotated, override

import polars as pl
from csv_read.generic import CSVReader, Schema
from csv_read.athena import OMOPVocabulariesV5
from rx_model import drug_classes as dc
from rx_model import hierarchy as h
from utils.exceptions import (
    ForeignNodeCreationError,
    UnmappedSourceConceptError,
)
from utils.logger import LOGGER


class SourceTable[IdS: pl.DataFrame | None](CSVReader[IdS], ABC):
    """
    Abstract class for reading BuildRxE input tables in CSV/TSV format.


    Attributes:
     TABLE_SCHEMA: Schema for the table.
        TABLE_COLUMNS: Ordered sequence of columns to keep from the table.
    """


class DrugConceptStage(SourceTable[None]):
    TABLE_SCHEMA: Schema = {
        "concept_code": pl.Utf8,
        "concept_name": pl.Utf8,
        "concept_class_id": pl.Utf8,
        "vocabulary_id": pl.Utf8,
        "source_concept_class_id": pl.Utf8,
        "possible_excipient": pl.Null,  # NOTE: not implemented yet
        "valid_start_date": pl.Date,
        "valid_end_date": pl.Date,
        "invalid_reason": pl.Utf8,
    }

    TABLE_COLUMNS: list[str] = [
        "concept_code",
        "concept_name",
        "concept_class_id",
        "vocabulary_id",
        "source_concept_class_id",
        # "possible_excipient",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
    ]

    @staticmethod
    def date_to_yyyymmdd(colname: str) -> pl.Expr:
        return (
            pl.col(colname).dt.year() * 1_00_00
            + pl.col(colname).dt.month() * 1_00
            + pl.col(colname).dt.day()
        )

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: None = None
    ) -> pl.LazyFrame:
        del valid_concepts
        return frame.select(
            pl.all().exclude("valid_start_date", "valid_end_date"),
            # Make dates YYYYMMDD integers
            valid_start_date=self.date_to_yyyymmdd("valid_start_date"),
            valid_end_date=self.date_to_yyyymmdd("valid_end_date"),
        )


class DSStage(SourceTable[pl.DataFrame]):
    type dss_strength_tuple = tuple[float | None, dc.PseudoUnit]

    TABLE_SCHEMA: Schema = {
        "drug_concept_code": pl.Utf8,
        "ingredient_concept_code": pl.Utf8,
        "amount_value": pl.Float64,
        "amount_unit": pl.Utf8,
        "numerator_value": pl.Float64,
        "numerator_unit": pl.Utf8,
        "denominator_value": pl.Float64,
        "denominator_unit": pl.Utf8,
        "box_size": pl.UInt16,
    }

    TABLE_COLUMNS: list[str] = [
        "drug_concept_code",
        "ingredient_concept_code",
        # Use matching order for strength tuple
        *dc.ForeignStrength._fields,
    ]

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: pl.DataFrame | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError("Valid concepts must be provided for DSStage.")

        return frame.filter(
            pl.col("drug_concept_code").is_in(valid_concepts["concept_code"]),
            pl.col("ingredient_concept_code").is_in(
                valid_concepts["concept_code"]
            ),
        )


class RelationshipToConcept(SourceTable[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "concept_code_1": pl.Utf8,
        "vocabulary_id_1": pl.Utf8,
        "concept_id_2": pl.UInt32,
        "precedence": pl.UInt8,
        "conversion_factor": pl.Float64,
    }

    TABLE_COLUMNS: list[str] = list(TABLE_SCHEMA.keys())

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: pl.DataFrame | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError(
                "Valid concepts must be provided for "
                "RelationshipToConceptStage."
            )

        return frame.filter(
            pl.col("concept_code_1").is_in(valid_concepts["concept_code"]),
            pl.col("vocabulary_id_1").is_in(valid_concepts["vocabulary_id"]),
            pl.col("precedence").is_null() | (pl.col("precedence") <= 1),
        )


class InternalRelationshipStage(SourceTable[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "concept_code_1": pl.Utf8,
        "concept_code_2": pl.Utf8,
    }

    TABLE_COLUMNS: list[str] = list(TABLE_SCHEMA.keys())

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: pl.DataFrame | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError(
                "Valid concepts must be provided for InternalRelationshipStage."
            )

        return frame.filter(
            pl.col("concept_code_1").is_in(valid_concepts["concept_code"]),
            pl.col("concept_code_2").is_in(valid_concepts["concept_code"]),
        )


class PCSStage(SourceTable[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "pack_concept_code": pl.Utf8,
        "drug_concept_code": pl.Utf8,
        "amount": pl.UInt16,
        "box_size": pl.UInt16,
    }
    TABLE_COLUMNS: list[str] = list(TABLE_SCHEMA.keys())

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: pl.DataFrame | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError("Valid concepts must be provided for PCSStage.")

        return frame.filter(
            pl.col("pack_concept_code").is_in(valid_concepts["concept_code"]),
            pl.col("drug_concept_code").is_in(valid_concepts["concept_code"]),
        )


class BuildRxEInput:
    """
    Class to read and prepare BuildRxE input data for evaluation.
    """

    def __init__(
        self,
        data_path: Path,
        athena_vocab: OMOPVocabulariesV5,
        delimiter: str = "\t",
        quote_char: str | None = None,
    ) -> None:
        self.data_path: Path = data_path

        self.logger: logging.Logger = LOGGER.getChild(self.__class__.__name__)

        # Remember the target hierarchy
        self.target_hierarchy: OMOPVocabulariesV5 = athena_vocab

        # Initiate containers
        self.source_atoms: h.Atoms[dc.ConceptCodeVocab] = h.Atoms(self.logger)
        self.rx_atoms: h.Atoms[dc.ConceptId] = athena_vocab.atoms
        self.translator: h.NodeTranslator = h.NodeTranslator(
            rx_atoms=self.rx_atoms, logger=self.logger
        )
        self.pseudo_units: list[dc.PseudoUnit] = []
        self.drug_nodes: list[
            dc.DrugNode[dc.ConceptCodeVocab, dc.Strength | None]
        ] = []

        # Read and prepare data
        self.logger.info(
            f"Starting processing of BuildRxE input tables from {data_path}"
        )

        self.dcs: DrugConceptStage = DrugConceptStage(
            path=data_path / "drug_concept_stage.tsv",
            delimiter=delimiter,
            quote_char=quote_char,
        )

        # Load valid concepts and populate the storages
        vocabs = self.dcs.collect()["vocabulary_id"].unique()
        assert len(vocabs) == 1
        self.load_valid_concepts()

        self.rtcs: RelationshipToConcept = RelationshipToConcept(
            data_path / "relationship_to_concept.tsv",
            reference_data=self.dcs.collect().select(
                "concept_code", "vocabulary_id"
            ),
        )

        self.translator.populate_from_frame(
            frame=self.rtcs.collect(),
            pseudo_units=self.pseudo_units,
        )

        self.ir: InternalRelationshipStage = InternalRelationshipStage(
            path=data_path / "internal_relationship_stage.tsv",
            reference_data=self.dcs.collect().select("concept_code"),
            delimiter=delimiter,
            quote_char=quote_char,
        )

        self.dss: DSStage = DSStage(
            data_path / "ds_stage.tsv",
            reference_data=self.dcs.collect().select("concept_code"),
        )

        # WARN: temporarily cleaning up all concepts with 0 in amount_value
        # There is actually a valid use case for this for drug packs, but we are
        # not processing them yet.
        zero_amounts = self.dss.collect().filter(pl.col("amount_value") == 0)
        if len(zero_amounts) > 0:
            self.logger.warning(
                f"Found {len(zero_amounts)} drugs with 0 amount_value. "
                "These will be ignored for now."
            )
        self.dss.anti_join(
            zero_amounts,
            left_on="drug_concept_code",
            right_on="drug_concept_code",
        )

        # WARN: temporarily cleaning up all concepts with box_size
        boxed_drugs = self.dss.collect().filter(
            pl.col("box_size").is_not_null()
        )
        if len(boxed_drugs) > 0:
            self.logger.warning(
                f"Found {len(boxed_drugs)} drugs with box_size. "
                "These will be ignored for now."
            )
        self.dcs.anti_join(
            boxed_drugs,
            left_on="concept_code",
            right_on="drug_concept_code",
        )

        # WARN: temporarily cleaning up all pack_concepts
        pcs = PCSStage(
            data_path / "pc_stage.tsv",
            reference_data=self.dcs.collect().select("concept_code"),
        )
        if len(pcs.collect()) > 0:
            self.logger.warning(
                f"Found {len(pcs.collect())} pack_concepts. "
                "These will be ignored for now."
            )
        self.dcs.anti_join(
            pcs.collect(),
            left_on="concept_code",
            right_on="pack_concept_code",
        )

    def load_valid_concepts(self) -> None:
        """
        Load valid concepts from DrugConceptStage and populate source_atoms.

        Populates self.source_atoms with valid concepts from DrugConceptStage,
        and registers all units as pseudo-units.
        """

        atom_concepts = (
            self.dcs.collect()
            .filter(
                pl.col("concept_class_id").is_in([
                    "Ingredient",
                    "Dose Form",
                    "Brand Name",
                    "Supplier",
                    "Unit",
                ])
            )
            .select(
                "concept_code",
                "vocabulary_id",
                "concept_name",
                "concept_class_id",
            )
        )

        self.logger.info(f"Loaded {len(atom_concepts)} valid concepts.")

        # Units must be excluded, as they are actually pseudo-units
        self.source_atoms.add_from_frame(
            atom_concepts.filter(pl.col("concept_class_id") != "Unit")
        )

        # Register all units as pseudo-units
        self.pseudo_units += atom_concepts.filter(
            pl.col("concept_class_id") == "Unit",
        )["concept_code"].to_list()

    def prepare_drug_nodes(
        self, crash_on_error: bool = False
    ) -> Generator[dc.ForeignNodePrototype, None, None]:
        """
        Build Node prototypes using the DSStage and InternalRelationshipStage
        data.
        """
        ir = self.ir.collect()
        dcs = self.dcs.collect()

        # First, get the unique attribute data
        drug_products = (
            dcs.filter(pl.col("concept_class_id") == "Drug Product")
            .join(
                # Exclude Drug Products having explicit mappings in RTC
                other=self.rtcs.collect(),
                left_on=["concept_code", "vocabulary_id"],
                right_on=["concept_code_1", "vocabulary_id_1"],
                how="anti",
            )
            .select("concept_code", "vocabulary_id")
        )

        for attr_class in ["Dose Form", "Brand Name", "Supplier"]:
            ir_of_attr = ir.join(
                other=dcs.filter(pl.col("concept_class_id") == attr_class),
                left_on="concept_code_2",
                right_on="concept_code",
                how="semi",
            )

            field_name = attr_class.lower().replace(" ", "_")
            try:
                drug_products = (
                    drug_products.join(
                        other=ir_of_attr,
                        left_on="concept_code",
                        right_on="concept_code_1",
                        how="left",
                        validate="1:1",  # TODO: Make this an external QA check
                    )
                    .with_columns(**{
                        f"{field_name}_code": pl.col("concept_code_2"),
                    })
                    .drop("concept_code_2")
                )
            except pl.exceptions.ComputeError as e:
                if crash_on_error:
                    raise e
                self.logger.error(
                    f"Error while validating uniqueness of {attr_class} data "
                    f"for Drug Products: {e}"
                )

        row: Annotated[tuple[str, ...], 5]
        for row in drug_products.iter_rows():
            vocab = row[1]
            drug_product_id = dc.ConceptCodeVocab(row[0], vocab)
            A = self.source_atoms
            attributes = [
                A.dose_form.get(dc.ConceptCodeVocab(row[2], vocab)),
                A.brand_name.get(dc.ConceptCodeVocab(row[3], vocab)),
                A.supplier.get(dc.ConceptCodeVocab(row[4], vocab)),
            ]

            yield dc.ForeignNodePrototype(
                drug_product_id,
                self.get_concept_strength(drug_product_id),
                *attributes,  # pyright: ignore[reportArgumentType]
            )

    def map_to_rxn(self) -> dict[dc.ConceptCodeVocab, list[dc.ConceptId]]:
        """
        Map the generated nodes to RxNorm concepts.
        """

        result: dict[dc.ConceptCodeVocab, list[dc.ConceptId]] = {}

        # 2 billion is conventionally used for loval concept IDs
        def new_concept_id():
            two_bill = 2_000_000_000
            while True:
                yield dc.ConceptId(two_bill)
                two_bill += 1

        cid_counter = new_concept_id()
        for node in self.prepare_drug_nodes(crash_on_error=False):
            translated_nodes = self.translator.translate_node(
                node, lambda: next(cid_counter)
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
                        f"Node {node.identifier} could not be mapped to "
                        f"RxNorm: {e}"
                    )
                    continue

                visitor = h.DrugNodeFinder(
                    option, self.target_hierarchy.hierarchy, self.logger
                )
                visitor.start_search()
                try:
                    node_result = visitor.get_search_results()
                except NotImplementedError:
                    # This is expected for now
                    self.logger.warning(
                        f"At least one valid Node for {node.identifier} could "
                        f"not be disambiguated."
                    )
                else:
                    result.setdefault(node.identifier, []).extend(
                        node.identifier for node in node_result.values()
                    )
        return result

    def get_concept_strength(
        self, drug_id: dc.ConceptCodeVocab
    ) -> list[dc.BoundForeignStrength]:
        """
        Extract strength combinations for a given drug concept.
        """
        strength_data = (
            self.dss.collect()
            .filter(pl.col("drug_concept_code") == drug_id.concept_code)
            .select(pl.all().exclude("drug_concept_code"))
        )

        ingredient_concept_code: str
        strength_combinations: list[dc.BoundForeignStrength] = []
        for ingredient_concept_code, *strength in strength_data.iter_rows():
            id = dc.ConceptCodeVocab(
                ingredient_concept_code, drug_id.vocabulary_id
            )
            try:
                ingredient = self.source_atoms.ingredient[id]
            except KeyError:
                raise ForeignNodeCreationError(
                    f"Ingredient with code {ingredient_concept_code} not found "
                    f"for drug {drug_id}."
                )
            strength_combinations.append((
                ingredient,
                dc.ForeignStrength._make(strength),
            ))

        return strength_combinations
