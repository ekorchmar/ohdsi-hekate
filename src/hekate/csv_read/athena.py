"""
Contains implementations to read CSV data from Athena OMOP CDM Vocabularies
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple, override

import polars as pl  # For type hinting and schema definition
from csv_read.generic import CSVReader, Schema
from rx_model import drug_classes as dc
from rx_model import hierarchy as h
from rx_model.exception import RxConceptCreationError
from utils.classes import SortedTuple
from utils.constants import (
    ALL_CONCEPT_RELATIONSHIP_IDS,
    DEFINING_ATTRIBUTE_RELATIONSHIP,
    STRENGTH_CONFIGURATIONS,
)
from utils.logger import LOGGER


class StrengthTuple(NamedTuple):
    ingredient_concept_id: int
    strength: h.UnboundStrength


class _StrengthDataRow(NamedTuple):
    drug_concept_id: int
    ingredient_concept_id: int
    amount_value: float
    amount_unit_concept_id: int
    numerator_value: float
    numerator_unit_concept_id: int
    denominator_value: float
    denominator_unit_concept_id: int
    amount_only: bool
    liquid_concentration: bool
    liquid_quantity: bool
    gas_concentration: bool
    # TODO: box_size and other strength data


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

    def get_strength_data(
        self, drug_ids: pl.Series
    ) -> dict[int, list[StrengthTuple]]:
        """
        Get the strength data slice for each drug concept.

        This does not validate the adherence of the concept to a particular
        strength data shape. This should be done either as a follow-up
        integrity check or in a separate method.

        The concept will be rejected only if it fails to match any of the
        valid configurations, or if it's entries match multiple configurations.

        Returns:
            A dictionary with drug concepts as keys and strength entries as
            values. A strength entry is a tuple with an integer concept_id and a
            variant of strength data, in shape of `SolidStrength`,
            `LiquidQuantity`, or `LiquidConcentration`.
        """
        strength_data: dict[int, list[StrengthTuple]] = {}
        strength_df = self.strength.data().join(
            other=drug_ids.unique().to_frame(name="drug_concept_id"),
            on="drug_concept_id",
        )

        # We want to be very strict about the strength data, so we will
        # look for explicit data shapes
        for label, expression in STRENGTH_CONFIGURATIONS.items():
            strength_df = strength_df.with_columns(**{label: expression})
        confirmed_drugs: pl.Series = self.filter_strength_chunk(strength_df)
        strength_df = strength_df.filter(
            pl.col("drug_concept_id").is_in(confirmed_drugs)
        )

        def get_unit(concept_id: int) -> dc.Unit:
            return self.atoms.unit[dc.ConceptId(concept_id)]

        # While we try to filter the data in advance, edge cases may
        # slip through.
        failed_concept_ids: list[int] = []

        for row in strength_df.iter_rows():
            row = _StrengthDataRow(*row)

            # Pick a Strength variant based on the determined configuration
            strength: h.UnboundStrength
            try:
                match True:
                    case row.amount_only:
                        strength = dc.SolidStrength(
                            amount_value=row.amount_value,
                            amount_unit=get_unit(row.amount_unit_concept_id),
                        )
                    case row.liquid_concentration:
                        strength = dc.LiquidConcentration(
                            numerator_value=row.numerator_value,
                            numerator_unit=get_unit(
                                row.numerator_unit_concept_id
                            ),
                            denominator_unit=get_unit(
                                row.denominator_unit_concept_id
                            ),
                        )
                    case row.liquid_quantity:
                        strength = dc.LiquidQuantity(
                            numerator_value=row.numerator_value,
                            numerator_unit=get_unit(
                                row.numerator_unit_concept_id
                            ),
                            denominator_value=row.denominator_value,
                            denominator_unit=get_unit(
                                row.denominator_unit_concept_id
                            ),
                        )
                    case row.gas_concentration:
                        strength = dc.GaseousPercentage(
                            numerator_value=row.numerator_value,
                            numerator_unit=get_unit(
                                row.numerator_unit_concept_id
                            ),
                        )
                    case _:
                        # Should be unreachable
                        raise ValueError(
                            f"Wrong configuration for {row.drug_concept_id}"
                        )
            except RxConceptCreationError as e:
                self.logger.error(
                    f"Failed to create strength data for {row.drug_concept_id}"
                    f": {e}"
                )
                failed_concept_ids.append(row.drug_concept_id)
                continue

            strength_data.setdefault(row.drug_concept_id, []).append(
                StrengthTuple(row.ingredient_concept_id, strength)
            )

            # TODO: save strength data to self.strengths
            # self.strengths.add_strength(row.ingredient_concept_id, strength)

        if failed_concept_ids:
            self.filter_out_bad_concepts(
                pl.Series(failed_concept_ids, dtype=pl.UInt32),
                "Strength_Creation",
                f"{len(failed_concept_ids):,} drug concepts had failed "
                "strength data creation",
            )
        else:
            self.logger.info("All strength data was successfully created")

        return strength_data

    def filter_strength_chunk(self, strength_chunk: pl.DataFrame) -> pl.Series:
        """
        Apply integrity checks to a strength chunk. Intended to be only called
        from `filter_invalid_strength_configurations` method. This method will
        return a pl.Series of only passing concept_ids.
        """

        # Find drugs that match no known configuration
        # NOTE: This is probably redundant after initial validation
        invalid_mask = (
            strength_chunk.select(
                *STRENGTH_CONFIGURATIONS.keys()
            ).sum_horizontal()
        ) == 0

        if n_invalid := invalid_mask.sum():
            self.logger.error(
                f"{n_invalid} drug concepts have invalid strength data and "
                f"will be excluded from the processing"
            )
            strength_chunk = strength_chunk.filter(~invalid_mask)
        else:
            self.logger.info("No invalid strength configurations found")

        # Find drugs that match more than one configuration over rows
        collapsed_df = (
            strength_chunk.select(
                "drug_concept_id", *STRENGTH_CONFIGURATIONS.keys()
            )
            .group_by("drug_concept_id")
            .max()  # T > F
        )
        muliple_match_mask = (
            collapsed_df.select(
                *STRENGTH_CONFIGURATIONS.keys()
            ).sum_horizontal()
            > 1
        )

        if n_unmatched := muliple_match_mask.sum():
            self.logger.error(
                f"{n_unmatched} drug concepts have ambiguously structured "
                f"strength data and will be excluded from the processing"
            )
            multiple_match_ids = collapsed_df["drug_concept_id"][
                muliple_match_mask
            ]
            strength_chunk = strength_chunk.filter(
                ~pl.col("drug_concept_id").is_in(multiple_match_ids)
            )
        else:
            self.logger.info("No ambiguous strength configurations found")

        # Filter out ambiguous denominator values
        # NOTE: this check is not important for the current release,
        # but it's future-proofing
        drugs_with_denom_counts = (
            strength_chunk.filter(
                pl.col("denominator_unit_concept_id").is_not_null()
            )
            .group_by("drug_concept_id")
            .agg(
                pl.struct(
                    pl.col("denominator_value"),
                    pl.col("denominator_unit_concept_id"),
                ).n_unique()
            )
        )
        struct_column_name = drugs_with_denom_counts.columns[1]
        ambiguous_denom_mask = drugs_with_denom_counts[struct_column_name] > 1

        if n_ambiguous_denom := ambiguous_denom_mask.sum():
            self.logger.error(
                f"{n_ambiguous_denom} drug concepts have ambiguous "
                f"denominator values and will be excluded from the processing"
            )
            ambiguous_denom_ids = drugs_with_denom_counts["drug_concept_id"][
                ambiguous_denom_mask
            ]
            strength_chunk = strength_chunk.filter(
                ~pl.col("drug_concept_id").is_in(ambiguous_denom_ids)
            )
        else:
            self.logger.info("No ambiguous denominator values found")

        return strength_chunk["drug_concept_id"]

    def __init__(self, vocab_download_path: Path):
        self.logger: logging.Logger = LOGGER.getChild(self.__class__.__name__)

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
        self.filter_deprecated_units_in_strength()
        self.filter_non_ingredient_in_strength()
        self.filter_invalid_strength_configurations()
        # TODO: filter gases with percentage sum exceeding 100

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
        cdf_nodes: list[int] = []

        cdf_concept_id: int
        self.logger.info("Finding Dose Forms for Clinical Drug Forms")
        cdf_dose_form: dict[int, dc.DoseForm[dc.ConceptId]] = {}
        cdf_to_df = self.get_class_relationships(
            class_id_1="Clinical Drug Form",
            class_id_2="Dose Form",
            relationship_id="RxNorm has dose form",
        )
        concept_id_idx = cdf_to_df.columns.index("concept_id")
        concept_id_target_idx = cdf_to_df.columns.index("concept_id_target")
        for row in cdf_to_df.iter_rows():
            cdf_concept_id = row[concept_id_idx]
            dose_form = self.atoms.dose_form[row[concept_id_target_idx]]
            cdf_dose_form[cdf_concept_id] = dose_form

        self.logger.info("Finding Ingredients for Clinical Drug Forms")
        cdf_ingredient: dict[int, list[dc.Ingredient[dc.ConceptId]]] = {}
        cdf_to_ing = self.get_class_relationships(
            class_id_1="Clinical Drug Form",
            class_id_2="Ingredient",
            relationship_id="RxNorm has ing",
        )
        concept_id_idx = cdf_to_ing.columns.index("concept_id")
        concept_id_target_idx = cdf_to_ing.columns.index("concept_id_target")
        for row in cdf_to_ing.iter_rows():
            cdf_concept_id = row[concept_id_idx]
            ingredient = self.atoms.ingredient[row[concept_id_target_idx]]
            cdf_ingredient.setdefault(cdf_concept_id, []).append(ingredient)

        self.logger.info("Creating Clinical Drug Forms")
        cdf_concepts: pl.Series = self.concept.data().filter(
            pl.col("concept_class_id") == "Clinical Drug Form"
        )["concept_id"]
        valid_cdf_count = 0

        cdf_ds_ing_count = self.get_strength_counts(cdf_concepts)

        cdf_no_df: list[int] = []
        cdf_no_ing: list[int] = []
        cdf_ing_mismatch: list[int] = []
        for cdf_concept_id in cdf_concepts:
            if (dose_form := cdf_dose_form.get(cdf_concept_id)) is None:
                cdf_no_df.append(cdf_concept_id)
                self.logger.warning(
                    f"Dose Form absent for Clinical Drug Form {cdf_concept_id}"
                )

            elif (ingreds := cdf_ingredient.get(cdf_concept_id)) is None:
                cdf_no_ing.append(cdf_concept_id)
                self.logger.warning(
                    f"Ingredients absent for Clinical Drug Form "
                    f"{cdf_concept_id}"
                )

            elif not cdf_ds_ing_count.get(cdf_concept_id, 0) == len(ingreds):
                cdf_ing_mismatch.append(cdf_concept_id)
                self.logger.warning(
                    f"Ingredient count mismatch for Clinical Drug Form "
                    f"{cdf_concept_id}"
                )

            else:
                try:
                    cdf = dc.ClinicalDrugForm(
                        identifier=dc.ConceptId(cdf_concept_id),
                        dose_form=dose_form,
                        ingredients=SortedTuple(ingreds),
                    )
                except RxConceptCreationError as e:
                    self.logger.error(
                        f"Failed to create Clinical Drug Form {cdf_concept_id}"
                        f": {e}"
                    )
                    continue
                valid_cdf_count += 1
                node_idx = self.hierarchy.add_clinical_drug_form(cdf)
                cdf_nodes.append(node_idx)

        self.logger.info(
            f"Processed {valid_cdf_count:,} Clinical Drug Forms out of "
            f"{len(cdf_concepts):,} possible."
        )

        bad_cdf = pl.Series(
            cdf_no_df + cdf_no_ing + cdf_ing_mismatch, dtype=pl.UInt32
        )
        if len(bad_cdf):
            self.filter_out_bad_concepts(
                bad_cdf,
                "CDF_Processing",
                f"{len(bad_cdf):,} Clinical Drug Forms had failed "
                "integrity checks",
            )
        else:
            self.logger.info("All Clinical Drug Forms passed integrity checks")

        return cdf_nodes

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
            self.filter_out_bad_concepts(
                complex_pi_as_ing["concept_id"],
                "PI_as_Ing",
                f"Found {len(complex_pi_as_ing):,} drug concepts that treat "
                f"Precise Ingredients as Ingredients",
            )
        else:
            self.logger.info(
                "No drug concepts that treat Precise Ingredients as "
                "Ingredients found"
            )

    def filter_out_bad_concepts(
        self, bad_concepts: pl.Series, reason_short: str, reason_full: str
    ) -> None:
        """
        Filter out concepts and their relationships from the tables.

        Args:
            bad_concepts: Polars Series with `concept_id` values to filter out.
            reason_short: Short reason for filtering out the concepts. Will be
                used for structuring the log messages and reports.
            reason_long: Reason for filtering out the concepts. Will be used for
                logging and/or reporting.
        """
        logger = self.logger.getChild(reason_short)
        logger.warning(reason_full)
        bad_concepts_df = bad_concepts.to_frame(name="concept_id")

        logger.info("Including all descendants of bad concepts")
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

        logger.info("Removing from the concept table")
        self.concept.reader.anti_join(bad_concepts_df, on=["concept_id"])

        logger.info("Removing from relationship table (left)")
        self.relationship.reader.anti_join(
            bad_concepts_df,
            left_on=["concept_id_1"],
            right_on=["concept_id"],
        )
        logger.info("Removing from relationship table (right)")
        self.relationship.reader.anti_join(
            bad_concepts_df,
            left_on=["concept_id_2"],
            right_on=["concept_id"],
        )

        logger.info("Removing from ancestor table (left)")
        self.ancestor.reader.anti_join(
            bad_concepts_df,
            left_on=["ancestor_concept_id"],
            right_on=["concept_id"],
        )

        logger.info("Removing from ancestor table (right)")
        self.ancestor.reader.anti_join(
            bad_concepts_df,
            left_on=["descendant_concept_id"],
            right_on=["concept_id"],
        )

        # Most of these checks are performed before the strength table
        # is materialized, and we don't want to materialize it just for this
        # operation.
        try:
            strength = self.strength
        except AttributeError:
            return

        logger.info("Removing from strength table (left)")
        strength.reader.anti_join(
            bad_concepts_df,
            left_on=["drug_concept_id"],
            right_on=["concept_id"],
        )
        logger.info("Removing from strength table (right)")
        strength.reader.anti_join(
            bad_concepts_df,
            left_on=["ingredient_concept_id"],
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
                self.filter_out_bad_concepts(
                    concept_to_multiple_valid,
                    "Multiple_" + concept_class.replace(" ", "_"),
                    f"Found {len(concept_to_multiple_valid):,} drug concepts "
                    f"with multiple valid {concept_class} attributes",
                )
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
                self.filter_out_bad_concepts(
                    concept_has_only_invalid,
                    "No_" + concept_class.replace(" ", "_"),
                    f"Found {len(concept_has_only_invalid):,} drug concepts "
                    f"with only invalid {concept_class} attributes",
                )
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
        cdc_nodes: list[int] = []

        self.logger.info("Finding Ingredients for Clinical Drug Components")

        # We could stick to DRUG_STRENGTH only, but we need to validate
        # the data
        tuples_to_check = [
            ("Ingredient", "RxNorm has ing", "Ing"),
            ("Precise Ingredient", "Has precise ing", "PI"),
        ]

        attrs: dict[str, pl.DataFrame] = {}
        for class_id, relationship_id, short in tuples_to_check:
            self.logger.info(f"Finding {class_id} for Clinical Drug Components")
            cdc_to_attr = self.get_class_relationships(
                class_id_1="Clinical Drug Comp",
                class_id_2=class_id,
                relationship_id=relationship_id,
            )
            cdc_to_mult = (
                cdc_to_attr.group_by("concept_id")
                .count()
                .filter(pl.col("count") > 1)
            )["concept_id"]
            if len(cdc_to_mult):
                self.filter_out_bad_concepts(
                    cdc_to_mult,
                    "CDC_Multiple_" + short,
                    f"Found {len(cdc_to_mult):,} Clinical Drug Components with "
                    f"multiple of {class_id} attributes",
                )
                cdc_to_attr = cdc_to_attr.filter(
                    ~pl.col("concept_id").is_in(cdc_to_mult)
                )
            else:
                self.logger.info(
                    f"No Clinical Drug Components with multiple of {class_id} "
                    f"attributes found"
                )

            attrs[short] = cdc_to_attr.select("concept_id", "concept_id_target")

        cdc_frame = (
            attrs["Ing"]
            .rename({"concept_id_target": "ingredient_concept_id"})
            .join(
                other=attrs["PI"].rename({
                    "concept_id_target": "precise_ingredient_concept_id"
                }),
                on="concept_id",
                how="left",
            )
        )

        # As we are working with CDC, only one strength entry is expected
        # per concept
        # NOTE: This check is semantically redundant, as we already filter
        # multiple ingredients per CONCEPT_RELATIONSHIP data, but there
        # are no guarantees of consistency.
        cdc_ds_ing_count = self.get_strength_counts(cdc_frame["concept_id"])
        multiple_strength = pl.Series(
            k for k, v in cdc_ds_ing_count.items() if v > 1
        )
        if len(multiple_strength):
            self.filter_out_bad_concepts(
                multiple_strength,
                "CDC_Multiple_Strength",
                f"Found {len(multiple_strength):,} Clinical Drug Components "
                "with multiple strength entries",
            )
            cdc_frame = cdc_frame.filter(
                ~pl.col("concept_id").is_in(multiple_strength)
            )
        else:
            self.logger.info(
                "No Clinical Drug Components with multiple strength entries "
                "found"
            )

        strength_data = self.get_strength_data(cdc_frame["concept_id"])
        cdc_frame = cdc_frame.filter(
            pl.col("concept_id").is_in(
                pl.Series(strength_data.keys(), dtype=pl.UInt32)
            )
        )

        cdc_ingredient_mismatch: list[int] = []
        cdc_bad_ingredient: list[int] = []
        cdc_bad_precise_ingredient: list[int] = []
        cdc_failed: list[int] = []
        for row in cdc_frame.iter_rows():
            concept_id: int = row[0]
            ingredient_concept_id: int = row[1]
            precise_ingredient_concept_id: int | None = row[2]
            (str_tuple,) = strength_data[concept_id]
            ds_ingredient_concept_id: int = str_tuple.ingredient_concept_id
            strength: h.UnboundStrength = str_tuple.strength

            # I really hope this check is redundant
            if ingredient_concept_id != ds_ingredient_concept_id:
                self.logger.error(
                    f"Ingredient mismatch for Clinical Drug Component "
                    f"{concept_id}: {ingredient_concept_id} != "
                    f"{ds_ingredient_concept_id}"
                )
                cdc_ingredient_mismatch.append(concept_id)

            try:
                ingredient = self.atoms.ingredient[
                    dc.ConceptId(ingredient_concept_id)
                ]
            except KeyError:
                self.logger.error(
                    f"Ingredient {ingredient_concept_id} not found for "
                    f"Clinical Drug Component {concept_id}"
                )
                cdc_bad_ingredient.append(concept_id)
                continue

            if (picid := precise_ingredient_concept_id) is not None:
                possible_pi = self.atoms.precise_ingredient.get(ingredient, [])
                possible_identifiers = [pi.identifier for pi in possible_pi]
                if dc.ConceptId(picid) not in possible_identifiers:
                    self.logger.error(
                        f"Precise Ingredient {picid} is not a valid Precise "
                        f"Ingredient for Ingredient {ingredient_concept_id}"
                    )
                    cdc_bad_precise_ingredient.append(concept_id)
                    continue
                else:
                    precise_ingredient = possible_pi[
                        possible_identifiers.index(dc.ConceptId(picid))
                    ]
            else:
                precise_ingredient = None

            try:
                cdc = dc.ClinicalDrugComponent(
                    identifier=dc.ConceptId(concept_id),
                    ingredient=ingredient,
                    precise_ingredient=precise_ingredient,
                    strength=strength,
                )
            except RxConceptCreationError as e:
                self.logger.error(
                    f"Failed to create Clinical Drug Component {concept_id}"
                    f": {e}"
                )
                cdc_failed.append(concept_id)
                continue

            node_idx: int = self.hierarchy.add_clinical_drug_component(cdc)
            cdc_nodes.append(node_idx)

        if len(cdc_ingredient_mismatch):
            self.filter_out_bad_concepts(
                pl.Series(cdc_ingredient_mismatch, dtype=pl.UInt32),
                "CDC_Ing_Mismatch",
                f"{len(cdc_ingredient_mismatch):,} Clinical Drug Components "
                f"had ingredient mismatches between DRUG_STRENGTH and "
                "CONCEPT_RELATIONSHIP",
            )
        else:
            self.logger.info(
                "All Clinical Drug Components have the same ingredient in "
                "DRUG_STRENGTH and CONCEPT_RELATIONSHIP"
            )

        if len(cdc_bad_ingredient):
            self.filter_out_bad_concepts(
                pl.Series(cdc_bad_ingredient, dtype=pl.UInt32),
                "CDC_Bad_Ing",
                f"{len(cdc_bad_ingredient):,} Clinical Drug Components had "
                "bad Ingredients",
            )
        else:
            self.logger.info(
                "All Clinical Drug Components have valid Ingredients"
            )

        if len(cdc_bad_precise_ingredient):
            self.filter_out_bad_concepts(
                pl.Series(cdc_bad_precise_ingredient, dtype=pl.UInt32),
                "CDC_Bad_PI",
                f"{len(cdc_bad_precise_ingredient):,} Clinical Drug Components "
                "had bad Precise Ingredients",
            )
        else:
            self.logger.info(
                "All Clinical Drug Components have valid Precise Ingredients"
            )

        if len(cdc_failed):
            self.filter_out_bad_concepts(
                pl.Series(cdc_failed, dtype=pl.UInt32),
                "CDC_Failed",
                f"{len(cdc_failed):,} Clinical Drug Components had failed "
                "creation",
            )
        else:
            self.logger.info(
                "All Clinical Drug Components were successfully created"
            )

        return cdc_nodes

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

            self.filter_out_bad_concepts(
                drug_strength_noning["drug_concept_id"],
                "Non_Ing_Strength",
                msg,
            )
            self.strength.reader.anti_join(
                drug_strength_noning, on=["drug_concept_id"]
            )
        else:
            self.logger.info(
                "No drug concepts that treat non-Ingredients as Ingredients "
                "in DRUG_STRENGTH found"
            )

    def filter_deprecated_units_in_strength(self):
        """
        Remove drugs that have deprecated units in DRUG_STRENGTH
        """

        self.logger.info("Filtering out deprecated units in DRUG_STRENGTH")

        # Exhausive list of valid unit ids
        valid_units = self.concept.data().filter(
            pl.col("concept_class_id") == "Unit",
            pl.col("standard_concept") == "S",
        )["concept_id"]

        unit_fields = [
            "amount_unit_concept_id",
            "numerator_unit_concept_id",
            "denominator_unit_concept_id",
        ]

        bad_units_expr = False
        for field in unit_fields:
            bad_units_expr |= ~pl.col(field).is_in(valid_units)

        bad_drugs = (self.strength.data().filter(bad_units_expr))[
            "drug_concept_id"
        ].unique()

        if len(bad_drugs):
            self.filter_out_bad_concepts(
                bad_drugs,
                "Deprecated_Unit",
                f"Found {len(bad_drugs):,} drug concepts with deprecated "
                "units in DRUG_STRENGTH",
            )
        else:
            self.logger.info("All units in DRUG_STRENGTH are valid")

    def filter_invalid_strength_configurations(self):
        """
        Filter out drug concepts with invalid strength configurations
        """
        # First, define the valid configurations
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
            # Comps and Drugs can have either amount concentration
            (
                pl.col("concept_class_id").is_in([
                    "Clinical Drug Comp",
                    "Branded Drug Comp",
                    "Clinical Drug",
                    "Branded Drug",
                ])
                & (
                    STRENGTH_CONFIGURATIONS["amount_only"]
                    | STRENGTH_CONFIGURATIONS["liquid_concentration"]
                    | STRENGTH_CONFIGURATIONS["gas_concentration"]
                )
            )
            |
            # Quant Drug classes can have only have liquid quantity
            (
                pl.col("concept_class_id").is_in([
                    "Quant Clinical Drug",
                    "Quant Branded Drug",
                ])
                & STRENGTH_CONFIGURATIONS["liquid_quantity"]
            )
        )

        invalid_strength = strength_with_class.filter(~valid_strength)

        if len(invalid_strength):
            # invalid_strength.write_csv(
            #     Path("reference") / "source_errors" / "invalid_strength.csv"
            # )
            invalid_drugs = invalid_strength["drug_concept_id"].unique()
            self.filter_out_bad_concepts(
                invalid_drugs,
                "Malformed_Strength",
                f"Found {len(invalid_strength):,} invalid strength "
                f"configurations for {len(invalid_drugs):,} drug concepts",
            )
        else:
            self.logger.info("All strength configurations are validated")
