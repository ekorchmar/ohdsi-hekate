"""
Contains implementations to read CSV data from Athena OMOP CDM Vocabularies
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import polars as pl  # For type hinting and schema definition

from csv_read.generic import CSVReader, Schema
from rx_model import drug_classes as dc
from rx_model import hierarchy as h
from utils.classes import SortedTuple
from utils.constants import ALL_CONCEPT_RELATIONSHIP_IDS
from utils.logger import LOGGER


class OMOPTable[FilterArg](ABC):
    """
    Abstract class to read Athena OMOP CDM Vocabularies
    """

    TABLE_SCHEMA: Schema
    TABLE_COLUMNS: list[str]
    reader: CSVReader
    path: Path

    @abstractmethod
    def table_filter(
        self, frame: pl.LazyFrame, filter_arg: FilterArg | None = None
    ) -> pl.LazyFrame:
        """
        Filter function to apply to the table.
        """

    def __init__(
        self,
        path: Path,
        logger: logging.Logger,
        filter_arg: FilterArg | None = None,
    ):
        super().__init__()
        self.path = path
        self.logger: logging.Logger = logger.getChild(self.__class__.__name__)

        self.logger.info(f"Reading {path.name}")
        self.reader = CSVReader(
            path=self.path,
            schema=self.TABLE_SCHEMA,
            line_filter=self.table_filter,
            filter_arg=filter_arg,
        )

    def data(self) -> pl.DataFrame:
        """
        Run the reader and return the data.
        """
        return self.reader.collect()


class ConceptTable(OMOPTable[None]):
    TABLE_SCHEMA: Schema = {
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
    TABLE_COLUMNS: list[str] = [
        "concept_id",
        "concept_name",
        # "domain_id", Made redundant by class
        "vocabulary_id",
        "concept_class_id",
        # "standard_concept",  # Determined by class
        "concept_code",
        "valid_start_date",
        # "valid_end_date",  # Known for valid concepts
        "invalid_reason",
    ]

    @override
    def table_filter(
        self, frame: pl.LazyFrame, filter_arg: None = None
    ) -> pl.LazyFrame:
        return (
            # TODO: test if this is faster than using the .is_in() method
            # Use .explain():
            # https://www.statology.org/how-to-use-explain-understand-lazyframe-query-optimization-polars/
            frame.filter(
                # Invalid reason is needed later
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
            .select(self.TABLE_COLUMNS)
        )


class RelationshipTable(OMOPTable[pl.Series]):
    TABLE_SCHEMA: Schema = {
        "concept_id_1": pl.UInt32,
        "concept_id_2": pl.UInt32,
        "relationship_id": pl.Utf8,
        "valid_start_date": pl.UInt32,
        "valid_end_date": pl.UInt32,
        "invalid_reason": pl.Utf8,
    }
    TABLE_COLUMNS: list[str] = [
        "concept_id_1",
        "concept_id_2",
        "relationship_id",
        "valid_start_date",
        # "valid_end_date",  # Known for valid relationships
        "invalid_reason",
    ]

    @override
    def table_filter(
        self, frame: pl.LazyFrame, filter_arg: pl.Series | None = None
    ) -> pl.LazyFrame:
        concept = filter_arg
        if concept is None:
            raise ValueError("Concept filter argument is required.")

        return frame.filter(
            # TODO: Hunt for valid relations to invalid targets
            pl.col("invalid_reason").is_null(),
            pl.col("relationship_id").is_in(ALL_CONCEPT_RELATIONSHIP_IDS),
            # TODO: Parametrize these joins; maybe it's not worth the
            # performance hit
            pl.col("concept_id_1").is_in(concept),
            pl.col("concept_id_2").is_in(concept),
        ).select(self.TABLE_COLUMNS)


class StrengthTable(OMOPTable[pl.Series]):
    TABLE_SCHEMA: Schema = {
        "drug_concept_id": pl.UInt32,
        "ingredient_concept_id": pl.UInt32,
        "amount_value": pl.Float64,
        "amount_unit_concept_id": pl.UInt32,
        "numerator_value": pl.Float64,
        "numerator_unit_concept_id": pl.UInt32,
        "denominator_value": pl.Float64,
        "denominator_unit_concept_id": pl.UInt32,
        "box_size": pl.UInt32,
        "valid_start_date": pl.UInt32,
        "valid_end_date": pl.UInt32,
        "invalid_reason": pl.Utf8,
    }
    TABLE_COLUMNS: list[str] = [
        "drug_concept_id",
        "ingredient_concept_id",
        "amount_value",
        "amount_unit_concept_id",
        "denominator_value",
        "denominator_unit_concept_id",
        # "box_size",  # Not used for now
        # NOTE: DRUG_STRENGTH implicitly contains only valid entries
        #
        # "valid_start_date",
        # "valid_end_date",
        # "invalid_reason",
    ]

    @override
    def table_filter(
        self, frame: pl.LazyFrame, filter_arg: pl.Series | None = None
    ) -> pl.LazyFrame:
        ingredients = filter_arg
        if ingredients is None:
            raise ValueError("Ingredient filter argument is required.")

        return frame.filter(
            pl.col("invalid_reason").is_null(),
            # Redundant
            # pl.col("drug_concept_id").is_in(concept["concept_id"]),
            pl.col("ingredient_concept_id").is_in(ingredients),
        ).select(self.TABLE_COLUMNS)


class OMOPVocabulariesV5:
    """
    Class to read Athena OMOP CDM Vocabularies
    """

    def get_class_relationships(
        self, class_id_1: str, class_id_2: str, relationship_id: str
    ) -> pl.DataFrame:
        """
        Get relationships of a defined type between concepts of two classes.
        """
        concept = self.concept.data()
        relationship = self.relationship.data().filter(
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
            .select(
                [
                    *concept.columns,
                    *map(lambda x: f"{x}_target", concept.columns),
                ]
            )
        )

        return joined

    def get_strength_counts(self, drug_ids: pl.Series) -> dict[int, int]:
        """
        Get the counts of ingredients for each drug concept.
        """
        counts_df = (
            self.strength.data()
            .join(
                other=drug_ids.to_frame(name="drug_concept_id"),
                left_on="drug_concept_id",
                right_on="drug_concept_id",
            )
            .group_by("drug_concept_id")
            .count()
            .rename({"count": "ingredient_count"})
        )

        return {row[0]: row[1] for row in counts_df.iter_rows()}

    def __init__(self, vocab_download_path: Path):
        self.logger: logging.Logger = LOGGER.getChild("VocabV5")

        # Initiate hierarchy containers
        self.atoms: h.Atoms[dc.ConceptId] = h.Atoms()
        self.strengths: h.KnownStrength[dc.ConceptId] = h.KnownStrength()
        self.hierarchy: h.RxHierarchy[dc.ConceptId] = h.RxHierarchy()

        # Vocabulary table readers
        self.concept: ConceptTable = ConceptTable(
            path=vocab_download_path / "CONCEPT.csv",
            logger=self.logger,
        )

        self.relationship: RelationshipTable = RelationshipTable(
            path=vocab_download_path / "CONCEPT_RELATIONSHIP.csv",
            logger=self.logger,
            filter_arg=self.concept.data()["concept_id"],
        )

        self.strength: StrengthTable = StrengthTable(
            path=vocab_download_path / "DRUG_STRENGTH.csv",
            logger=self.logger,
            filter_arg=self.concept.data()["concept_id"],
        )

        # TODO: filter implicitly deprecated concepts

        # Process the drug classes from the simplest to the most complex
        self.process_atoms()
        self.process_precise_ingredients()
        self.process_clinical_drug_forms()

    def process_atoms(self) -> None:
        """
        Process atom concepts with known concept data.
        """
        # Populate atoms with known concepts
        self.logger.info(
            "Processing atomic concepts (Ingredient, Dose Form, etc.)"
        )
        rxn_atoms: pl.DataFrame = (
            self.concept.data()
            .select(
                [
                    "concept_id",
                    "concept_name",
                    "concept_class_id",
                ]
            )
            .filter(
                pl.col("concept_class_id").is_in(
                    [
                        "Ingredient",
                        # "Precise Ingredient",  # Needs linking to Ingredient
                        "Brand Name",
                        "Dose Form",
                        "Supplier",
                        "Unit",
                    ]
                )
            )
        )
        self.atoms.add_from_frame(rxn_atoms)

        # Add ingredients as roots to hierarchy
        self.logger.info("Adding Ingredients to the hierarchy")
        for ingredient in self.atoms.ingredient.values():
            self.hierarchy.add_root(ingredient)

    def process_precise_ingredients(self) -> None:
        """
        Process Precise Ingredients and link them to Ingredients
        """
        ing_to_precise = self.get_class_relationships(
            class_id_1="Precise Ingredient",
            class_id_2="Ingredient",
            relationship_id="Form of",  # Maps to?
        )

        self.logger.info("Processing Precise Ingredients")
        column_names = ing_to_precise.columns
        concept_id_target_idx = column_names.index("concept_id_target")
        concept_id_idx = column_names.index("concept_id")
        concept_name_idx = column_names.index("concept_name")
        for row in ing_to_precise.iter_rows():
            ingredient = self.atoms.ingredient[row[concept_id_target_idx]]
            precise_identifier: int = row[concept_id_idx]
            precise_name: str = row[concept_name_idx]
            self.atoms.add_precise_ingredient(
                dc.PreciseIngredient(
                    identifier=dc.ConceptId(precise_identifier),
                    concept_name=precise_name,
                    invariant=ingredient,
                )
            )

    def process_clinical_drug_forms(self) -> None:
        """
        Process Clinical Drug Forms and link them to Dose Forms and Ingredients
        """
        self.logger.info("Processing Clinical Drug Forms")

        cdc_concept_id: int
        self.logger.info("Finding Dose Forms for Clinical Drug Forms")
        cdc_dose_form: dict[int, dc.DoseForm[dc.ConceptId]] = {}
        cdc_to_df = self.get_class_relationships(
            class_id_1="Clinical Drug Form",
            class_id_2="Dose Form",
            relationship_id="RxNorm has dose form",
        )
        concept_id_idx = cdc_to_df.columns.index("concept_id")
        concept_id_target_idx = cdc_to_df.columns.index("concept_id_target")
        for row in cdc_to_df.iter_rows():
            cdc_concept_id = row[concept_id_idx]
            dose_form = self.atoms.dose_form[row[concept_id_target_idx]]
            cdc_dose_form[cdc_concept_id] = dose_form

        self.logger.info("Finding Ingredients for Clinical Drug Forms")
        cdc_ingredient: dict[int, list[dc.Ingredient[dc.ConceptId]]] = {}
        cdc_to_ing = self.get_class_relationships(
            class_id_1="Clinical Drug Form",
            class_id_2="Ingredient",
            relationship_id="RxNorm has ing",
        )
        concept_id_idx = cdc_to_ing.columns.index("concept_id")
        concept_id_target_idx = cdc_to_ing.columns.index("concept_id_target")
        for row in cdc_to_ing.iter_rows():
            cdc_concept_id = row[concept_id_idx]
            ingredient = self.atoms.ingredient[row[concept_id_target_idx]]
            cdc_ingredient.setdefault(cdc_concept_id, []).append(ingredient)

        # WARN: RxE concepts may have Precise Ingredient in place of Ingredient,
        # but this IS an error, as evidenced by DRUG_STRENGTH entries. We are
        # discarding those Clinical Drug Forms.

        self.logger.info("Creating Clinical Drug Forms")
        cdc_concepts = self.concept.data().filter(
            pl.col("concept_class_id") == "Clinical Drug Form"
        )["concept_id"]
        valid_cdc_count = 0

        cdc_ds_ing_count = self.get_strength_counts(cdc_concepts)

        for cdc_concept_id in cdc_concepts:
            dose_form = cdc_dose_form.get(cdc_concept_id)
            ingreds = cdc_ingredient.get(cdc_concept_id)

            if dose_form is None:
                self.logger.warning(
                    f"Dose Form absent for Clinical Drug Form {cdc_concept_id}"
                )

            elif ingreds is None:
                self.logger.warning(
                    f"Ingredients absent for Clinical Drug Form "
                    f"{cdc_concept_id}"
                )

            elif not cdc_ds_ing_count.get(cdc_concept_id, 0) == len(ingreds):
                self.logger.warning(
                    f"Ingredient count mismatch for Clinical Drug Form "
                    f"{cdc_concept_id}"
                )

            else:
                valid_cdc_count += 1
                self.hierarchy.add_clinical_drug_form(
                    dc.ClinicalDrugForm(
                        identifier=dc.ConceptId(cdc_concept_id),
                        dose_form=dose_form,
                        ingredients=SortedTuple(ingreds),
                    )
                )

        self.logger.info(
            f"Processed {valid_cdc_count} Clinical Drug Forms with "
            f"Dose Forms and Ingredients out of {len(cdc_concepts)} "
            f"possible."
        )
