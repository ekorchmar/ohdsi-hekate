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
from utils.constants import (
    ALL_CONCEPT_RELATIONSHIP_IDS,
    DEFINING_ATTRIBUTE_RELATIONSHIP,
    PERCENT_CONCEPT_ID,
)
from utils.logger import LOGGER


class OMOPTable[FilterArg](ABC):
    """
    Abstract class to read Athena OMOP CDM Vocabularies

    Attributes:
        TABLE_SCHEMA: Schema for the table.
        TABLE_COLUMNS: Ordered sequence of columns to keep from the table.
        reader: CSVReader instance to read the table.

    """

    TABLE_SCHEMA: Schema
    TABLE_COLUMNS: list[str]

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
        self.path: Path = path
        self.logger: logging.Logger = logger.getChild(self.__class__.__name__)

        self.logger.info(f"Reading {path.name}")
        self.reader: CSVReader = CSVReader(
            path=self.path,
            schema=self.TABLE_SCHEMA,
            keep_columns=self.TABLE_COLUMNS,
            line_filter=self.table_filter,
            filter_arg=filter_arg,
        )

    def data(self) -> pl.DataFrame:
        """
        Run the reader and return the data.
        """
        return self.reader.collect()

    def materialize(self) -> None:
        """
        Materialize the lazy frame into a DataFrame.
        """
        self.reader.materialize()


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
        "standard_concept",
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
                # Invalid reason is needed for filtering, so we keep it
                # pl.col("invalid_reason").is_null(),
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
        # "invalid_reason",
    ]

    @override
    def table_filter(
        self, frame: pl.LazyFrame, filter_arg: pl.Series | None = None
    ) -> pl.LazyFrame:
        concept = filter_arg
        if concept is None:
            raise ValueError("Concept filter argument is required.")

        return frame.filter(
            # Only care about valid internal relationships
            pl.col("invalid_reason").is_null(),
            pl.col("relationship_id").is_in(ALL_CONCEPT_RELATIONSHIP_IDS),
            # TODO: Parametrize these joins; maybe it's not worth the
            # performance hit
            pl.col("concept_id_1").is_in(concept),
            pl.col("concept_id_2").is_in(concept),
        )


class StrengthTable(OMOPTable[pl.DataFrame]):
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
        "numerator_value",
        "numerator_unit_concept_id",
        "denominator_value",
        "denominator_unit_concept_id",
        # "box_size",  # Not used for now
    ]

    @override
    def table_filter(
        self, frame: pl.LazyFrame, filter_arg: pl.DataFrame | None = None
    ) -> pl.LazyFrame:
        known_valid_drugs = filter_arg
        if known_valid_drugs is None:
            raise ValueError("Valid drugs as filter argument is required.")

        return frame.filter(
            pl.col("invalid_reason").is_null(),
            pl.col("drug_concept_id").is_in(known_valid_drugs["concept_id"]),
        )


class AncestorTable(OMOPTable[pl.Series]):
    TABLE_SCHEMA: Schema = {
        "ancestor_concept_id": pl.UInt32,
        "descendant_concept_id": pl.UInt32,
        "min_levels_of_separation": pl.UInt32,
        "max_levels_of_separation": pl.UInt32,
    }
    TABLE_COLUMNS: list[str] = [
        "ancestor_concept_id",
        "descendant_concept_id",
        # "min_levels_of_separation",
        # "max_levels_of_separation",
    ]

    @override
    def table_filter(
        self, frame: pl.LazyFrame, filter_arg: pl.Series | None = None
    ) -> pl.LazyFrame:
        concept = filter_arg
        if concept is None:
            raise ValueError("Concept filter argument is required.")

        # Only keep internal relationships
        return frame.filter(
            pl.col("descendant_concept_id").is_in(concept),
            pl.col("ancestor_concept_id").is_in(concept),
        )


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
            .select([
                *concept.columns,
                *map(lambda x: f"{x}_target", concept.columns),
            ])
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
            )["drug_concept_id"]
            .value_counts()
        )

        return {row[0]: row[1] for row in counts_df.iter_rows()}

    def get_strength_data(self, drug_ids: pl.Series) -> pl.DataFrame:
        """
        Get the strength data for each drug concept.
        """
        return self.strength.data().join(
            other=drug_ids.unique().to_frame(name="drug_concept_id"),
            on="drug_concept_id",
        )

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

        # Materialize the concept table to filter big tables early
        self.concept.materialize()
        all_concept_ids = self.concept.data()["concept_id"]

        self.relationship: RelationshipTable = RelationshipTable(
            path=vocab_download_path / "CONCEPT_RELATIONSHIP.csv",
            logger=self.logger,
            filter_arg=all_concept_ids,
        )

        self.ancestor: AncestorTable = AncestorTable(
            path=vocab_download_path / "CONCEPT_ANCESTOR.csv",
            logger=self.logger,
            filter_arg=all_concept_ids,
        )

        # Filter implicitly deprecated concepts
        self.filter_malformed_concepts()

        # Now there are much less concepts to process
        self.strength: StrengthTable = StrengthTable(
            path=vocab_download_path / "DRUG_STRENGTH.csv",
            logger=self.logger,
            filter_arg=self.concept.data(),
        )

        # Also use Drug Strength for filtering other tables
        self.filter_non_ingredient_in_strength()
        self.filter_invalid_strength_configurations()

        # Process the drug classes from the simplest to the most complex
        self.process_atoms()
        self.process_precise_ingredients()
        cdf_nodes: list[int] = self.process_clinical_drug_forms()
        cdc_nodes: list[int] = self.process_clinical_drug_comps()

        del cdf_nodes, cdc_nodes

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
            .select([
                "concept_id",
                "concept_name",
                "concept_class_id",
            ])
            .filter(
                pl.col("concept_class_id").is_in([
                    "Ingredient",
                    # "Precise Ingredient",  # Needs linking to Ingredient
                    "Brand Name",
                    "Dose Form",
                    "Supplier",
                    "Unit",
                ])
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

    def process_clinical_drug_forms(self) -> list[int]:
        """
        Process Clinical Drug Forms and link them to Dose Forms and Ingredients
        """
        self.logger.info("Processing Clinical Drug Forms")
        # Save node indices for reuse by descending classes
        cdc_nodes: list[int] = []

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

        self.logger.info("Creating Clinical Drug Forms")
        cdc_concepts: pl.Series = self.concept.data().filter(
            pl.col("concept_class_id") == "Clinical Drug Form"
        )["concept_id"]
        valid_cdc_count = 0

        cdc_ds_ing_count = self.get_strength_counts(cdc_concepts)

        cdc_no_df: list[int] = []
        cdc_no_ing: list[int] = []
        cdc_ing_mismatch: list[int] = []
        for cdc_concept_id in cdc_concepts:
            if (dose_form := cdc_dose_form.get(cdc_concept_id)) is None:
                cdc_no_df.append(cdc_concept_id)
                self.logger.warning(
                    f"Dose Form absent for Clinical Drug Form {cdc_concept_id}"
                )

            elif (ingreds := cdc_ingredient.get(cdc_concept_id)) is None:
                cdc_no_ing.append(cdc_concept_id)
                self.logger.warning(
                    f"Ingredients absent for Clinical Drug Form "
                    f"{cdc_concept_id}"
                )

            elif not cdc_ds_ing_count.get(cdc_concept_id, 0) == len(ingreds):
                cdc_ing_mismatch.append(cdc_concept_id)
                self.logger.warning(
                    f"Ingredient count mismatch for Clinical Drug Form "
                    f"{cdc_concept_id}"
                )

            else:
                valid_cdc_count += 1
                node_idx = self.hierarchy.add_clinical_drug_form(
                    dc.ClinicalDrugForm(
                        identifier=dc.ConceptId(cdc_concept_id),
                        dose_form=dose_form,
                        ingredients=SortedTuple(ingreds),
                    )
                )
                cdc_nodes.append(node_idx)

        self.logger.info(
            f"Processed {valid_cdc_count:,} Clinical Drug Forms out of "
            f"{len(cdc_concepts):,} possible."
        )

        bad_cdc = pl.Series(
            cdc_no_df + cdc_no_ing + cdc_ing_mismatch, dtype=pl.UInt32
        )
        if len(bad_cdc):
            self.logger.warning(
                f"{len(bad_cdc):,} Clinical Drug Forms had failed "
                "integrity checks"
            )
            self.filter_out_bad_concepts(bad_cdc)

        return cdc_nodes

    def filter_malformed_concepts(self) -> None:
        """
        Filter out concepts with invalid or missing data. This first performs
        integrity checks and then removes the explicitly deprecated concepts.
        """
        self.logger.info("Filtering out malformed concepts")

        self.filter_precise_ingredient_as_ingredient()

        self.salvage_multiple_defining_attributes()

        # Remove explicitly deprecated concepts and their relations
        self.filter_explicitly_deprecated_concepts()

    def filter_precise_ingredient_as_ingredient(self):
        """
        Filter out drug concepts that treat Precise Ingredients as
        Ingredients
        """
        self.logger.info(
            "Filtering out drug concepts that treat Precise Ingredients as "
            "Ingredients"
        )
        prec_ing_concepts = (
            self.concept.data()
            .filter(
                pl.col("vocabulary_id") == "RxNorm",
                pl.col("concept_class_id") == "Precise Ingredient",
                pl.col("invalid_reason").is_null(),
            )
            .select(["concept_id"])
        )
        complex_drug_concepts = self.concept.data().filter(
            pl.col("standard_concept") == "S",
            ~(pl.col("concept_class_id") == "Ingredient"),
        )
        complex_pi_as_ing = (
            prec_ing_concepts.join(
                other=self.relationship.data().filter(
                    # NOTE: This relationship_id is reserved for Ingredients!
                    pl.col("relationship_id") == "RxNorm ing of",
                ),
                left_on="concept_id",
                right_on="concept_id_1",
                suffix="_relationship",
            )
            .join(
                other=complex_drug_concepts,
                left_on="concept_id_2",
                right_on="concept_id",
                suffix="_target",
            )
            .select(concept_id="concept_id_2")
        )

        if len(complex_pi_as_ing):
            self.logger.warning(
                f"Found {len(complex_pi_as_ing):,} drug concepts that treat "
                "Precise Ingredients as Ingredients"
            )
            self.filter_out_bad_concepts(complex_pi_as_ing["concept_id"])
        else:
            self.logger.info(
                "No drug concepts that treat Precise Ingredients as "
                "Ingredients found"
            )

    def filter_out_bad_concepts(self, bad_concepts: pl.Series) -> None:
        """
        Filter out concepts and their relationships from the tables.

        Args:
            bad_concepts: Polars Series with `concept_id` values to filter out.
        """
        bad_concepts_df = bad_concepts.to_frame(name="concept_id")

        self.logger.info("Including all descendants of bad concepts")
        bad_descendants_df = (
            self.ancestor.data()
            .join(
                other=bad_concepts_df,
                left_on="ancestor_concept_id",
                right_on="concept_id",
            )
            .select(concept_id="descendant_concept_id")
            .unique()
        )
        bad_concepts_df = pl.concat([bad_concepts_df, bad_descendants_df])

        self.logger.info("Removing from the concept table")
        self.concept.reader.anti_join(bad_concepts_df, on=["concept_id"])

        self.logger.info("Removing from relationship table (left)")
        self.relationship.reader.anti_join(
            bad_concepts_df,
            left_on=["concept_id_1"],
            right_on=["concept_id"],
        )
        self.logger.info("Removing from relationship table (right)")
        self.relationship.reader.anti_join(
            bad_concepts_df,
            left_on=["concept_id_2"],
            right_on=["concept_id"],
        )

        self.logger.info("Removing from ancestor table (left)")
        self.ancestor.reader.anti_join(
            bad_concepts_df,
            left_on=["ancestor_concept_id"],
            right_on=["concept_id"],
        )

        self.logger.info("Removing from ancestor table (right)")
        self.ancestor.reader.anti_join(
            bad_concepts_df,
            left_on=["descendant_concept_id"],
            right_on=["concept_id"],
        )

    def filter_explicitly_deprecated_concepts(self):
        """
        Filter out explicitly deprecated concepts
        """
        self.logger.info("Filtering out invalid concepts and their relations")
        invalid_concepts = (
            self.concept.data()
            .filter(pl.col("invalid_reason").is_not_null())
            .select("concept_id")
        )
        self.logger.info(
            f"Found {len(invalid_concepts):,} invalid concepts "
            f"out of {len(self.concept.data()):,}"
        )

        # NOTE: Do not use filter_out_bad_concepts() here, this is faster
        self.concept.reader.filter(pl.col("invalid_reason").is_null())
        self.relationship.reader.filter(
            ~(
                pl.col("concept_id_1").is_in(invalid_concepts["concept_id"])
                | pl.col("concept_id_2").is_in(invalid_concepts["concept_id"])
            )
        )

    def salvage_multiple_defining_attributes(self):
        """
        Salvage concepts that specify more than one defining attribute,
        only one of them being valid.
        """
        complex_drug_concepts = self.concept.data().filter(
            pl.col("standard_concept") == "S",
            ~(pl.col("concept_class_id") == "Ingredient"),
        )

        for concept_class, rel in DEFINING_ATTRIBUTE_RELATIONSHIP.items():
            self.logger.info(
                f"Salvaging concepts that specify more than one "
                f"{concept_class} attribute"
            )

            # Find concepts with multiple defining attributes
            rel_to_attribute = (
                complex_drug_concepts.join(
                    other=self.relationship.data().filter(
                        pl.col("relationship_id") == rel,
                    ),
                    left_on="concept_id",
                    right_on="concept_id_1",
                    suffix="_relationship",
                )
                .join(
                    other=self.concept.data().filter(
                        pl.col("concept_class_id") == concept_class,
                    ),
                    left_on="concept_id_2",
                    right_on="concept_id",
                    suffix="_target",
                )
                .select("concept_id", "concept_id_2", "invalid_reason_target")
            )

            # We want to catch only:
            # - Concepts with multiple valid attributes
            # - Concepts with exactly 0 valid attributes
            #     These would be misenterpreted as concepts of another class
            #     by Build_RxE. Hekate would reject them at a later stage still,
            #     but it's better to catch them early.

            # First, count with multiple valid attributes
            concept_to_multiple_valid = (
                rel_to_attribute.filter(
                    pl.col("invalid_reason_target").is_null()
                )["concept_id"]
                .value_counts()
                .filter(pl.col("count") > 1)
            )["concept_id"]

            if len(concept_to_multiple_valid):
                self.logger.warning(
                    f"Found {len(concept_to_multiple_valid):,} drug concepts "
                    f"with multiple valid {concept_class} attributes"
                )
                self.filter_out_bad_concepts(concept_to_multiple_valid)
            else:
                self.logger.info(
                    f"No drug concepts with multiple valid {concept_class} "
                    "attributes found"
                )

            # Second, find ones having only any amount of invalid attributes
            concept_has_valid = rel_to_attribute.filter(
                pl.col("invalid_reason_target").is_null()
            ).select("concept_id")
            concept_has_only_invalid = (
                rel_to_attribute.join(
                    concept_has_valid,
                    left_on="concept_id",
                    right_on="concept_id",
                    how="anti",
                )
            )["concept_id"].unique()

            if len(concept_has_only_invalid):
                self.logger.warning(
                    f"Found {len(concept_has_only_invalid):,} drug concepts "
                    f"with only invalid {concept_class} attributes"
                )
                self.filter_out_bad_concepts(concept_has_only_invalid)
            else:
                self.logger.info(
                    f"No drug concepts with only invalid {concept_class} "
                    "attributes found"
                )

            # Log the number of concepts salvaged
            # For this, find concepts with multiple relations
            # and substract the ones that were filtered out
            # NOTE: This is not critical for Hekate run, but we may want to
            # eventually export the QA data for Vocabularies to fix the upstream
            # source data.
            concept_to_multiple_relations = (
                rel_to_attribute["concept_id"]
                .value_counts()
                .filter(pl.col("count") > 1)
            )["concept_id"]
            diff = len(concept_to_multiple_relations) - len(
                concept_to_multiple_valid
            )
            if diff:
                self.logger.info(
                    f"Salvaged {diff:,} drug concepts with multiple "
                    f"{concept_class} attributes"
                )
            else:
                self.logger.info(
                    f"No drug concepts with multiple {concept_class} "
                    "attributes were salvaged"
                )

    def process_clinical_drug_comps(self) -> list[int]:
        """
        Process Clinical Drug Components and link them to Ingredients and
        Precise Ingredients (if available)
        """
        self.logger.info("Processing Clinical Drug Components")
        # Save node indices for reuse by descending classes
        cdf_nodes: list[int] = []

        self.logger.info("Finding Ingredients for Clinical Drug Components")

        # We could stick to DRUG_STRENGTH only, but we need to validate
        # the data
        cdf_to_ing = self.get_class_relationships(
            class_id_1="Clinical Drug Comp",
            class_id_2="Ingredient",
            relationship_id="RxNorm has ing",
        )
        cdf_to_mult = (
            cdf_to_ing.group_by("concept_id")
            .count()
            .filter(pl.col("count") > 1)
        )["concept_id"]
        if len(cdf_to_mult):
            self.logger.warning(
                f"Found {len(cdf_to_mult):,} Clinical Drug Components with "
                "multiple Ingredients"
            )
            self.filter_out_bad_concepts(cdf_to_mult)
            cdf_to_ing = cdf_to_ing.join(
                other=cdf_to_mult.to_frame(name="concept_id"),
                on="concept_id",
                how="anti",
            )

        # Find the Precise Ingredients for Clinical Drug Components
        cdf_to_precise = self.get_class_relationships(
            class_id_1="Clinical Drug Comp",
            class_id_2="Precise Ingredient",
            relationship_id="Has precise ing",
        )
        cdf_to_mult_pi = (
            cdf_to_precise["concept_id"]
            .value_counts()
            .filter(pl.col("count") > 1)
        )["concept_id"]

        if len(cdf_to_mult_pi):
            self.logger.warning(
                f"Found {len(cdf_to_mult_pi):,} Clinical Drug Components with "
                f"multiple Precise Ingredients"
            )
            self.filter_out_bad_concepts(cdf_to_mult_pi)
            cdf_to_precise = cdf_to_precise.join(
                other=cdf_to_mult_pi.to_frame(name="concept_id"),
                on="concept_id",
                how="anti",
            )

        strength_data = self.get_strength_data(cdf_to_ing["concept_id"])

        del cdf_nodes, strength_data
        raise NotImplementedError(
            "Finish implementing Clinical Drug Components"
        )

    def filter_non_ingredient_in_strength(self):
        """
        Filter out drug concepts that treat other drug classes
        (usually Precise Ingredients) as Ingredients in DRUG_STRENGTH entries
        """

        self.logger.info(
            "Filtering out drug concepts that specify non-Ingredients in "
            "DRUG_STRENGTH"
        )
        drug_strength_noning = (
            self.strength.data()
            .join(
                other=self.concept.data(),
                left_on="ingredient_concept_id",
                right_on="concept_id",
            )
            .filter(
                pl.col("concept_class_id") != "Ingredient",
            )
            .select("drug_concept_id", "concept_class_id")
        )

        if len(drug_strength_noning):
            msg = (
                f"Found {len(drug_strength_noning):,} drug concepts that "
                f"treat non-Ingredients as Ingredients in DRUG_STRENGTH, "
                f"by target:"
            )
            cls_counts = drug_strength_noning["concept_class_id"].value_counts()
            cls: str
            cnt: int
            for cls, cnt in cls_counts.iter_rows():
                msg += f"\n- {cnt:,} {cls} concepts"

            self.logger.warning(msg)
            self.filter_out_bad_concepts(
                drug_strength_noning["drug_concept_id"]
            )
            self.strength.reader.anti_join(
                drug_strength_noning, on=["drug_concept_id"]
            )
        else:
            self.logger.info(
                "No drug concepts that treat non-Ingredients as Ingredients "
                "in DRUG_STRENGTH found"
            )

    def filter_invalid_strength_configurations(self):
        """
        Filter out drug concepts with invalid strength configurations
        """
        # First, define the valid configurations
        # - Amount value and unit are present, rest of the fields are null
        amount_only = (
            pl.col("amount_value").is_not_null()
            & pl.col("amount_unit_concept_id").is_not_null()
            & pl.col("numerator_value").is_null()
            & pl.col("numerator_unit_concept_id").is_null()
            & pl.col("denominator_value").is_null()
            & pl.col("denominator_unit_concept_id").is_null()
        )
        # - Numerator value and unit are present, but denominator is unit only
        liquid_concentration = (
            pl.col("amount_value").is_null()
            & pl.col("amount_unit_concept_id").is_null()
            & pl.col("numerator_value").is_not_null()
            & pl.col("numerator_unit_concept_id").is_not_null()
            & pl.col("denominator_value").is_null()
            & pl.col("denominator_unit_concept_id").is_not_null()
        )
        # - Numerator and denominator values and units are present
        liquid_quantity = (
            pl.col("amount_value").is_null()
            & pl.col("amount_unit_concept_id").is_null()
            & pl.col("numerator_value").is_not_null()
            & pl.col("numerator_unit_concept_id").is_not_null()
            & pl.col("denominator_value").is_not_null()
            & pl.col("denominator_unit_concept_id").is_not_null()
        )
        # - Gases are weird. They can have numerator with percents exactly in
        #   numerator and no denominator data.
        gas_concentration = (
            pl.col("amount_value").is_null()
            & pl.col("amount_unit_concept_id").is_null()
            & pl.col("numerator_value").is_not_null()
            & (pl.col("numerator_unit_concept_id") == PERCENT_CONCEPT_ID)  # (%)
            & pl.col("denominator_value").is_null()
            & pl.col("denominator_unit_concept_id").is_null()
        )
        # - Quantified gases are even worse! They can have numerator value,
        #   percents numerator unit and denominator value with no unit.
        gas_quantity = (
            pl.col("amount_value").is_null()
            & pl.col("amount_unit_concept_id").is_null()
            & pl.col("numerator_value").is_not_null()
            & (pl.col("numerator_unit_concept_id") == PERCENT_CONCEPT_ID)  # (%)
            & pl.col("denominator_value").is_not_null()
            & pl.col("denominator_unit_concept_id").is_null()
        )

        # TODO: Box size variations, once we start using them

        strength_with_class = (
            self.strength.data()
            .join(
                other=self.concept.data(),
                left_on="drug_concept_id",
                right_on="concept_id",
            )
            .select(*StrengthTable.TABLE_COLUMNS, "concept_class_id")
        )

        # Filter configurations by class
        valid_strength: pl.Expr = (
            # WARN: There are CDF, BDF and Ingredient concept that specify
            # amount unit ONLY. This is not a valid configuration, but we will
            # let it slide for now, as their strength data is meaningless.
            pl.col("concept_class_id").is_in([
                "Clinical Drug Form",
                "Branded Drug Form",
                "Ingredient",
                # "Precise Ingredient",  # Always wrong
            ])
            |
            # Comps and Drugs can have either amount or liquid concentration
            (
                pl.col("concept_class_id").is_in([
                    "Clinical Drug Comp",
                    "Branded Drug Comp",
                    "Clinical Drug",
                    "Branded Drug",
                ])
                & (amount_only | liquid_concentration | gas_concentration)
            )
            |
            # Quant Drug classes can have any quantity of non-solid phase
            (
                pl.col("concept_class_id").is_in([
                    "Quant Clinical Drug",
                    "Quant Branded Drug",
                ])
                & (liquid_quantity | gas_quantity)
            )
        )

        invalid_strength = strength_with_class.filter(~valid_strength)

        if len(invalid_strength):
            # invalid_strength.write_csv(
            #     Path("reference") / "source_errors" / "invalid_strength.csv"
            # )
            invalid_drugs = invalid_strength["drug_concept_id"].unique()
            self.logger.warning(
                f"Found {len(invalid_strength):,} invalid strength "
                f"configurations for {len(invalid_drugs):,} drug concepts"
            )
            self.filter_out_bad_concepts(invalid_drugs)
            self.strength.reader.anti_join(
                invalid_drugs.to_frame(name="drug_concept_id"),
                on=["drug_concept_id"],
            )
        else:
            self.logger.info("All strength configurations are validated")
