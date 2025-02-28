"""
Contains implementations to read CSV data from Athena OMOP CDM Vocabularies
"""

import logging
from pathlib import Path

import polars as pl  # For type hinting and schema definition

from csv_read.generic import CSVReader, Schema
from rx_model import drug_classes as dc
from rx_model import hierarchy as h
from utils.constants import ALL_CONCEPT_RELATIONSHIP_IDS
from utils.logger import LOGGER


class OMOPVocabulariesV5:
    """
    Class to read Athena OMOP CDM Vocabularies
    """

    CONCEPT_SCHEMA: Schema = {
        "concept_id": pl.UInt32,
        "concept_name": pl.Utf8,
        "domain_id": pl.Utf8,
        "vocabulary_id": pl.Utf8,
        "concept_class_id": pl.Utf8,
        "standard_concept": pl.Utf8,
        "concept_code": pl.Utf8,
        # Athena dates are given as YYYYMMDD, and polars does not pick them up
        "valid_start_date": pl.UInt32,  # NB: Not pl.Date
        "valid_end_date": pl.UInt32,
        "invalid_reason": pl.Utf8,
    }
    CONCEPT_COLUMNS: list[str] = [
        "concept_id",
        "concept_name",
        # "domain_id", Made redundant by class
        "vocabulary_id",
        "concept_class_id",
        # "standard_concept",  # Determined by class
        "concept_code",
        "valid_start_date",
        # "valid_end_date",  # Known for valid concepts
    ]

    CONCEPT_RELATIONSHIP_SCHEMA: Schema = {
        "concept_id_1": pl.UInt32,
        "concept_id_2": pl.UInt32,
        "relationship_id": pl.Utf8,
        "valid_start_date": pl.UInt32,
        "valid_end_date": pl.UInt32,
        "invalid_reason": pl.Utf8,
    }
    CONCEPT_RELATIONSHIP_COLUMNS: list[str] = [
        "concept_id_1",
        "concept_id_2",
        "relationship_id",
        "valid_start_date",
        # "valid_end_date",  # Known for valid relationships
        # "invalid_reason",  # Not used
    ]

    def concept_filter(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        return (
            # TODO: test if this is faster than using the .is_in() method
            # Use .explain():
            # https://www.statology.org/how-to-use-explain-understand-lazyframe-query-optimization-polars/
            frame.filter(
                pl.col("invalid_reason").is_null(),
                (
                    (pl.col("domain_id") == "Drug")
                    | (pl.col("domain_id") == "Unit")
                ),
                (
                    (pl.col("vocabulary_id") == "RxNorm")
                    | (pl.col("vocabulary_id") == "RxNorm Extension")
                    | (pl.col("vocabulary_id") == "UCUM")
                ),
                (
                    # RxNorm atoms
                    (pl.col("concept_class_id") == "Ingredient")
                    | (pl.col("concept_class_id") == "Precise Ingredient")
                    | (pl.col("concept_class_id") == "Brand Name")
                    | (pl.col("concept_class_id") == "Dose Form")
                    |
                    # RxNorm Extension atoms
                    (pl.col("concept_class_id") == "Supplier")
                    |
                    # UCUM atoms
                    (pl.col("concept_class_id") == "Unit")
                    |
                    # RxNorm drug concepts
                    (pl.col("concept_class_id") == "Clinical Drug Comp")
                    | (pl.col("concept_class_id") == "Branded Drug Comp")
                    | (pl.col("concept_class_id") == "Clinical Drug Form")
                    | (pl.col("concept_class_id") == "Branded Drug Form")
                    | (pl.col("concept_class_id") == "Clinical Drug")
                    | (pl.col("concept_class_id") == "Branded Drug")
                    | (pl.col("concept_class_id") == "Quant Clinical Drug")
                    | (pl.col("concept_class_id") == "Quant Branded Drug")
                    # RxNorm Extension drug concepts (unused for now)
                    # | (pl.col("concept_class_id") == "Clinical Drug Box")
                    # | (pl.col("concept_class_id") == "Branded Drug Box")
                    # | (pl.col("concept_class_id") == "Quant Clinical Box")
                    # | (pl.col("concept_class_id") == "Quant Branded Box")
                    # | (pl.col("concept_class_id") == "Clinical Pack")
                    # | (pl.col("concept_class_id") == "Branded Pack")
                    # | (pl.col("concept_class_id") == "Clinical Pack Box")
                    # | (pl.col("concept_class_id") == "Branded Pack Box")
                ),
            )
            # Only subset of columns (in predictable order)
            .select(self.CONCEPT_COLUMNS)
        )

    def concept_relationship_filter(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        concept = self.concept_reader.data
        if concept is None:
            raise ValueError("Concept data must be read first!")

        return frame.filter(
            pl.col("invalid_reason").is_null(),
            pl.col("relationship_id").is_in(ALL_CONCEPT_RELATIONSHIP_IDS),
            # TODO: Parametrize these joins; maybe it's not worth the
            # performance hit
            pl.col("concept_id_1").is_in(concept["concept_id"]),
            pl.col("concept_id_2").is_in(concept["concept_id"]),
        ).select(self.CONCEPT_RELATIONSHIP_COLUMNS)

    @property
    def concept_data(self) -> pl.DataFrame:
        return self.concept_reader.collect()

    @property
    def relationship_data(self) -> pl.DataFrame:
        return self.relationship_reader.collect()

    @property
    def strength_data(self) -> pl.DataFrame:
        raise NotImplementedError("Not implemented yet")
        return self.strength_reader.collect()

    def get_class_relationships(
        self, class_id_1: str, class_id_2: str, relationship_id: str
    ) -> pl.DataFrame:
        """
        Get relationships of a defined type between concepts of two classes.
        """
        concept = self.concept_data
        relationship = self.relationship_data.filter(
            pl.col("relationship_id") == relationship_id
        )

        joined = (
            concept.filter(pl.col("concept_class_id") == class_id_1)
            .join(
                other=relationship,
                left_on="concept_id",
                right_on="concept_id_1",
                suffix="_relationship",
            )
            .join(
                other=concept.filter(pl.col("concept_class_id") == class_id_2),
                left_on="concept_id_2",
                right_on="concept_id",
                suffix="_target",
            )
            .rename({"concept_id_2": "concept_id_target"})
            .select([
                *self.CONCEPT_COLUMNS,
                *map(lambda x: f"{x}_target", self.CONCEPT_COLUMNS),
            ])
        )

        return joined

    def __init__(self, vocab_download_path: Path):
        self.logger: logging.Logger = LOGGER.getChild("VocabV5")

        # Initiate hierarchy containers
        self.atoms: h.Atoms[dc.ConceptId] = h.Atoms()
        self.strengths: h.KnownStrength[dc.ConceptId] = h.KnownStrength()
        self.hierarchy: h.RxHierarchy[dc.ConceptId] = h.RxHierarchy()

        self.vocab_download_path: Path = vocab_download_path

        # Concept
        self.logger.info("Reading CONCEPT.csv")
        concept_path = self.vocab_download_path / "CONCEPT.csv"

        self.concept_reader: CSVReader = CSVReader(
            path=concept_path,
            schema=self.CONCEPT_SCHEMA,
            line_filter=self.concept_filter,
        )

        # Populate atoms with known concepts
        self.logger.info("Processing atomic concepts (Ingredient, Dose Form, etc.)")
        rxn_atoms: pl.DataFrame = self.concept_data.select([
            "concept_id",
            "concept_name",
            "concept_class_id",
        ]).filter(
            pl.col("concept_class_id").is_in([
                "Ingredient",
                # "Precise Ingredient",  # Needs CONCEPT_RELATIONSHIP
                "Brand Name",
                "Dose Form",
                "Supplier",
                "Unit",
            ])
        )
        self.atoms.add_from_frame(rxn_atoms)

        # Concept Relationship
        logging.info("Reading CONCEPT_RELATIONSHIP.csv")
        concept_relationship_path = (
            self.vocab_download_path / "CONCEPT_RELATIONSHIP.csv"
        )

        self.relationship_reader: CSVReader = CSVReader(
            path=concept_relationship_path,
            schema=self.CONCEPT_RELATIONSHIP_SCHEMA,
            line_filter=self.concept_relationship_filter,
        )

        # We can now also process Precise Ingredients
        ing_to_precise = self.get_class_relationships(
            class_id_1="Precise Ingredient",
            class_id_2="Ingredient",
            relationship_id="Form of",  # Maps to?
        )

        for row in ing_to_precise.iter_rows(named=True):
            ingredient = self.atoms.ingredient[row["concept_id_target"]]
            precise_identifier: int = row["concept_id"]
            precise_name: str = row["concept_name"]
            self.atoms.add_precise_ingredient(
                dc.PreciseIngredient(
                    identifier=dc.ConceptId(precise_identifier),
                    concept_name=precise_name,
                    invariant=ingredient,
                )
            )
