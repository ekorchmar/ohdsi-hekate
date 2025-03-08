"""
Contains implementations to read TSV data from a file and transform it
into ForeignDrugNode objects for evaluation.
"""

import logging
from abc import ABC
from pathlib import Path
from typing import override
from collections.abc import Generator

import polars as pl
from csv_read.generic import CSVReader, Schema
from hekate.rx_model.drug_classes.generic import ConceptCodeVocab
from hekate.utils.classes import SortedTuple
from rx_model import drug_classes as dc
from rx_model import hierarchy as h

from utils.logger import LOGGER  # For type hinting and schema definition


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
    type dss_strength_tuple = tuple[float | None, h.PseudoUnit]

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
        "amount_value",
        "amount_unit",
        "numerator_value",
        "numerator_unit",
        "denominator_value",
        "denominator_unit",
        # "box_size"  # Unused for now
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
            # WARN: temporarily discard all data mentioning box_size
            pl.col("box_size").is_null(),
        ).select(pl.all().exclude("box_size"))


class RelationshipToConcept(SourceTable[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "concept_code_1": pl.Utf8,
        "vocabulary_id_1": pl.Utf8,
        "concept_id_2": pl.UInt32,
        "precedence": pl.UInt8,
        "conversion_factor": pl.Float64,
    }

    TABLE_COLUMNS: list[str] = list(TABLE_SCHEMA.keys())

    # TODO: Implement precedence. For now, discard all rows with precedence > 1
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
        "relationship_id": pl.UInt32,
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
                "Valid concepts must be provided for InternalRelationshipStage."
            )

        return frame.filter(
            pl.col("concept_code_1").is_in(valid_concepts["concept_code"]),
            pl.col("concept_code_2").is_in(valid_concepts["concept_code"]),
        )


class BuildRxEInput:
    """
    Class to read and prepare BuildRxE input data for evaluation.
    """

    def __init__(
        self,
        data_path: Path,
        rx_atoms: h.Atoms[dc.ConceptId],
        delimiter: str = "\\t",
        quote_char: str = '"',
    ) -> None:
        self.data_path: Path = data_path

        self.logger: logging.Logger = LOGGER.getChild(self.__class__.__name__)

        # Initiate containers
        self.source_atoms: h.Atoms[dc.ConceptCodeVocab] = h.Atoms()
        self.rx_atoms: h.Atoms[dc.ConceptId] = rx_atoms
        self.atom_mapper: h.AtomMapper = h.AtomMapper(
            atom_getter=self.rx_atoms.lookup_unknown,
            unit_storage=self.rx_atoms.unit,
        )
        self.pseudo_units: list[h.PseudoUnit] = []
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
            data_path / "relationship_to_concept_stage.tsv",
            reference_data=self.dcs.collect().select(
                "concept_code", "vocabulary_id"
            ),
        )

        self.atom_mapper.populate_from_frame(
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

    def load_valid_concepts(self) -> None:
        """
        Load valid concepts from DrugConceptStage and populate source_atoms.

        Populates self.source_atoms with valid concepts from DrugConceptStage,
        and registers all units as pseudo-units.
        """

        atom_concepts = self.dcs.collect().select(
            "concept_code", "vocabulary_id", "concept_name", "concept_class_id"
        )

        # Units must be excluded, as they are actually pseudo-units
        self.source_atoms.add_from_frame(
            atom_concepts.filter(pl.col("concept_class_id") != "Unit")
        )

        # Register all units as pseudo-units
        self.pseudo_units += atom_concepts.filter(
            pl.col("concept_class_id") == "Unit",
        )["concept_code"].to_list()

    def build_drug_nodes(
        self,
    ) -> Generator[
        dc.DrugNode[dc.ConceptCodeVocab, dc.Strength | None], None, None
    ]:
        """
        Build DrugNodes using the DSStage and InternalRelationshipStage data.
        """
        drug_products = self.dcs.collect().filter(
            pl.col("concept_class_id") == "Drug Product"
        )

        del drug_products
        raise NotImplementedError(
            "BuildRxEInput.build_drug_nodes not implemented"
        )

    def get_strength_combinations(
        self, drug_product_id: dc.ConceptCodeVocab
    ) -> Generator[
        SortedTuple[dc.BoundStrength[dc.ConceptCodeVocab, dc.Strength | None]],
        None,
        None,
    ]:
        """
        Get dc.strength combinations from the DSStage data.
        """
        strength_data = self.dss.collect().filter(
            pl.col("drug_concept_code") == drug_product_id.concept_code
        )

        # For drugs without strength data, just return the ingredients from
        # the IRS
        if len(strength_data) == 0:
            ingredient_codes = (
                self.ir.collect()
                .filter(
                    pl.col("concept_code_1") == drug_product_id.concept_code
                )
                .join(
                    self.dcs.collect(),
                    left_on="concept_code_2",
                    right_on="concept_code",
                )["concept_code", "vocabulary_id"]
            )

            strengthless: list[
                tuple[dc.Ingredient[dc.ConceptCodeVocab], None]
            ] = []
            for row in ingredient_codes.iter_rows():
                concept_code: str = row[0]
                vocab_id: str = row[1]
                ingredient = self.source_atoms.ingredient[
                    ConceptCodeVocab(concept_code, vocab_id)
                ]
                strengthless.append((ingredient, None))

            yield SortedTuple(strengthless)
            return

        # For drugs without strength data, implement :D
        raise NotImplementedError(
            "BuildRxEInput.get_strength_combinations not implemented"
        )
