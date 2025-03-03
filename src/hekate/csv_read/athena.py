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


class _StrengthTuple(NamedTuple):
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


# Type hint for a dictionary linking int concept_ids to indices in the hierarchy
# graph. This serves as a temporary cache to speed up the hierarchy building
type _TempNodeView = dict[int, int]


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

    def get_strength_data(
        self, drug_ids: pl.Series
    ) -> dict[int, list[_StrengthTuple]]:
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
        strength_data: dict[int, list[_StrengthTuple]] = {}
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
                _StrengthTuple(row.ingredient_concept_id, strength)
            )

            # TODO: save strength data to self.strengths
            # self.strengths.add_strength(row.ingredient_concept_id, strength)

        self.filter_out_bad_concepts(
            pl.Series(failed_concept_ids, dtype=pl.UInt32),
            "All strength data was successfully created",
            "Strength_Creation",
            f"{len(failed_concept_ids):,} drug concepts had failed "
            "strength data creation",
        )

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

        # Concepts that do not have a valid Ingredient ancestor are effectively
        # not participating in the hierarchy, so we need to avoid mapping to
        # them
        self.filter_orphaned_complex_drugs()

        # Filter implicitly deprecated concepts
        self.filter_precise_ingredient_as_ingredient()
        self.salvage_multiple_defining_attributes()

        # Remove explicitly deprecated concepts and their relations
        self.filter_explicitly_deprecated_concepts()

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
        cdf_nodes: _TempNodeView = self.process_clinical_drug_forms()
        cdc_nodes: _TempNodeView = self.process_clinical_drug_comps()
        bdf_nodes: _TempNodeView = self.process_branded_drug_forms(cdf_nodes)

        del cdc_nodes, bdf_nodes

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

    def process_clinical_drug_forms(self) -> _TempNodeView:
        """
        Process Clinical Drug Forms and link them to Dose Forms and Ingredients

        Returns:
            List of node indices for Clinical Drug Forms in the hierarchy
        """
        self.logger.info("Processing Clinical Drug Forms")
        # Save node indices for reuse by descending classes
        cdf_nodes: _TempNodeView = {}
        self.logger.info("Finding Dose Forms for Clinical Drug Forms")
        cdf_to_df = self.get_class_relationships(
            class_id_1="Clinical Drug Form",
            class_id_2="Dose Form",
            relationship_id="RxNorm has dose form",
        ).select("concept_id", dose_form_id="concept_id_target")

        cdf_concepts = (
            self.concept.data()
            .filter(pl.col("concept_class_id") == "Clinical Drug Form")
            .select("concept_id")
            .join(cdf_to_df, on="concept_id", how="left")
        )

        # Catch empty dose forms
        cdf_no_df = cdf_concepts.filter(pl.col("dose_form_id").is_null())
        self.filter_out_bad_concepts(
            cdf_no_df["concept_id"],
            "All Clinical Drug Forms have a Dose Form",
            "CDF_no_DF",
            f"{len(cdf_no_df):,} Clinical Drug Forms had no Dose Form",
        )
        if len(cdf_no_df):
            cdf_concepts = cdf_concepts.filter(
                pl.col("dose_form_id").is_not_null()
            )

        # Catch multiple dose forms for a single Clinical Drug Form
        cdf_mult_df = (
            cdf_to_df.group_by("concept_id").count().filter(pl.col("count") > 1)
        )
        self.filter_out_bad_concepts(
            cdf_mult_df["concept_id"],
            "All Clinical Drug Forms had a single Dose Form",
            "CDF_Mult_DF",
            f"{len(cdf_mult_df):,} Clinical Drug Forms had multiple Dose Forms",
        )
        if len(cdf_mult_df):
            cdf_concepts = cdf_concepts.join(
                cdf_mult_df, on="concept_id", how="anti"
            )

        self.logger.info("Finding Ingredients for Clinical Drug Forms")

        # From CONCEPT_RELATIONSHIP table
        cdf_to_ing_cr = self.get_class_relationships(
            class_id_1="Clinical Drug Form",
            class_id_2="Ingredient",
            relationship_id="RxNorm has ing",
        ).select(["concept_id", "concept_id_target"])

        cdf_ing_cr: dict[int, set[int]] = {}
        for row in cdf_to_ing_cr.iter_rows():
            cr_cdc_id: int = row[0]
            cr_ingredient_id: int = row[1]
            cdf_ing_cr.setdefault(cr_cdc_id, set()).add(cr_ingredient_id)

        # From DRUG_STRENGTH table
        cdf_to_ing_ds = (
            self.strength.data()
            .filter(pl.col("drug_concept_id").is_in(cdf_concepts["concept_id"]))
            .select(["drug_concept_id", "ingredient_concept_id"])
        )
        cdf_ing_ds: dict[int, set[int]] = {}
        for row in cdf_to_ing_ds.iter_rows():
            ds_cdc_id: int = row[0]
            ds_ingredient_id: int = row[1]
            cdf_ing_ds.setdefault(ds_cdc_id, set()).add(ds_ingredient_id)

        cdf_no_ing: list[int] = []
        cdf_ingredient_mismatch: list[int] = []
        cdf_bad_df: list[int] = []
        cdf_bad_ings: list[int] = []
        cdf_failed: list[int] = []
        for row in cdf_concepts.iter_rows():
            concept_id: int = row[0]
            dose_form_id: int = row[1]
            ingredients_ds = cdf_ing_ds.get(concept_id, set())
            ingredients_cr = cdf_ing_cr.get(concept_id, set())

            if len(ingredients_ds) == 0:
                self.logger.error(
                    f"Clinical Drug Form {concept_id} had no Ingredients"
                )
                cdf_no_ing.append(concept_id)
                continue

            if ingredients_ds != ingredients_cr:
                self.logger.error(
                    f"Clinical Drug Form {concept_id} had mismatched "
                    f"Ingredients between CONCEPT_RELATIONSHIP and "
                    f"DRUG_STRENGTH tables"
                )
                cdf_ingredient_mismatch.append(concept_id)
                continue

            missing_ingredient = None
            ingreds: list[dc.Ingredient[dc.ConceptId]] = []
            for ingredient_id in ingredients_ds:
                try:
                    ingredient = self.atoms.ingredient[
                        dc.ConceptId(ingredient_id)
                    ]
                except KeyError:
                    missing_ingredient = ingredient_id
                    break
                ingreds.append(ingredient)
            if missing_ingredient is not None:
                self.logger.error(
                    f"Clinical Drug Form {concept_id} had missing Ingredient "
                    f"{missing_ingredient}"
                )
                cdf_bad_ings.append(concept_id)
                continue

            try:
                dose_form = self.atoms.dose_form[dc.ConceptId(dose_form_id)]
            except KeyError:
                self.logger.error(
                    f"Clinical Drug Form {concept_id} had missing Dose Form "
                    f"{dose_form_id}"
                )
                cdf_bad_df.append(concept_id)
                continue

            try:
                cdf = dc.ClinicalDrugForm(
                    identifier=dc.ConceptId(concept_id),
                    dose_form=dose_form,
                    ingredients=SortedTuple(ingreds),
                )
            except RxConceptCreationError as e:
                self.logger.error(
                    f"Failed to create Clinical Drug Form {concept_id}: {e}"
                )
                cdf_failed.append(concept_id)
                continue

            node_idx = self.hierarchy.add_clinical_drug_form(cdf)
            cdf_nodes[concept_id] = node_idx

        self.filter_out_bad_concepts(
            pl.Series(cdf_no_ing, dtype=pl.UInt32),
            "All Clinical Drug Forms had Ingredients",
            "CDF_No_Ing",
            f"{len(cdf_no_ing):,} Clinical Drug Forms had no Ingredients",
        )

        self.filter_out_bad_concepts(
            pl.Series(cdf_ingredient_mismatch, dtype=pl.UInt32),
            "All Clinical Drug Forms had matching Ingredients",
            "CDF_Ing_Mismatch",
            f"{len(cdf_ingredient_mismatch):,} Clinical Drug Forms had "
            "mismatched Ingredients",
        )

        self.filter_out_bad_concepts(
            pl.Series(cdf_bad_df, dtype=pl.UInt32),
            "All Clinical Drug Forms had valid Dose Forms",
            "CDF_Bad_DF",
            f"{len(cdf_bad_df):,} Clinical Drug Forms had bad Dose Forms",
        )

        self.filter_out_bad_concepts(
            pl.Series(cdf_bad_ings, dtype=pl.UInt32),
            "All Clinical Drug Forms had valid Ingredients",
            "CDF_Bad_Ing",
            f"{len(cdf_bad_ings):,} Clinical Drug Forms had bad Ingredients",
        )

        self.filter_out_bad_concepts(
            pl.Series(cdf_failed, dtype=pl.UInt32),
            "All Clinical Drug Forms were successfully created",
            "CDF_Failed",
            f"{len(cdf_failed):,} Clinical Drug Forms had failed creation",
        )

        return cdf_nodes

    def filter_orphaned_complex_drugs(self):
        """
        Filter out complex drug concepts that have no valid ancestor.

        Drug concepts, when they participate in a concept set, are included
        based on their hierarchical relationship. If a drug concept has no
        valid Ingredient ancestor, it is effectively not participating in the
        hierarchy. Mapping to such a concept would be a mistake.
        """

        self.logger.info("Filtering out orphaned complex drug concepts")

        ingredient_descendants = (
            self.concept.data()
            .filter(
                pl.col("standard_concept") == "S",
                pl.col("concept_class_id") == "Ingredient",
            )
            .join(
                other=self.ancestor.data(),
                left_on="concept_id",
                right_on="ancestor_concept_id",
            )
            .select(concept_id="descendant_concept_id")
        )

        orphaned_complex_concepts = (
            self.concept.data()
            .filter(
                pl.col("standard_concept") == "S",
                pl.col("concept_class_id") != "Ingredient",
                pl.col("concept_class_id") != "Unit",
            )
            .join(
                other=ingredient_descendants,
                on="concept_id",
                how="anti",
            )
        )

        self.filter_out_bad_concepts(
            orphaned_complex_concepts["concept_id"],
            "All complex drug concepts have a valid Ingredient ancestor",
            "Orphaned_Complex",
            f"{len(orphaned_complex_concepts):,} complex drug concepts have "
            f"no ancestor Ingredient",
        )

        # NOTE: We should also check for broken hierarchy links in between the
        # complex drug concepts themselves, but it is not as important for
        # practical applications.

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

        self.filter_out_bad_concepts(
            complex_pi_as_ing["concept_id"],
            "No drug concepts that treat Precise Ingredients as "
            "Ingredients found",
            "PI_as_Ing",
            f"Found {len(complex_pi_as_ing):,} drug concepts that treat "
            f"Precise Ingredients as Ingredients",
        )

    def filter_out_bad_concepts(
        self,
        bad_concepts: pl.Series,
        message_ok: str,
        reason_short: str,
        reason_full: str,
    ) -> None:
        """
        Filter out concepts and their relationships from the tables.

        Args:
            bad_concepts: Polars Series with `concept_id` values to filter out.
            message_ok: Message to log if no bad concepts are found.
            reason_short: Short reason for filtering out the concepts. Will be
                used for structuring the log messages and reports.
            reason_long: Reason for filtering out the concepts. Will be used for
                logging and/or reporting.
        """
        logger = self.logger.getChild(reason_short)

        if len(bad_concepts) == 0:
            logger.info(message_ok)
            return

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

            self.filter_out_bad_concepts(
                concept_to_multiple_valid,
                message_ok="No drug concepts with multiple valid "
                f"{concept_class} attributes found",
                reason_short="Multiple_" + concept_class.replace(" ", "_"),
                reason_full=f"Found {len(concept_to_multiple_valid):,} drug "
                f"concepts with multiple valid {concept_class} attributes",
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

            self.filter_out_bad_concepts(
                concept_has_only_invalid,
                f"No drug concepts with only invalid {concept_class} "
                "attributes found",
                "No_" + concept_class.replace(" ", "_"),
                f"Found {len(concept_has_only_invalid):,} drug concepts "
                f"with only invalid {concept_class} attributes",
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

    def process_clinical_drug_comps(self) -> _TempNodeView:
        """
        Process Clinical Drug Components and link them to Ingredients and
        Precise Ingredients (if available)

        Returns:
            List of node indices for Clinical Drug Components in the hierarchy
        """
        self.logger.info("Processing Clinical Drug Components")
        # Save node indices for reuse by descending classes
        cdc_nodes: _TempNodeView = {}

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
            self.filter_out_bad_concepts(
                cdc_to_mult,
                f"No Clinical Drug Components with multiple of {class_id} "
                f"attributes found",
                "CDC_Multiple_" + short,
                f"Found {len(cdc_to_mult):,} Clinical Drug Components with "
                f"multiple of {class_id} attributes",
            )
            if len(cdc_to_mult):
                cdc_to_attr = cdc_to_attr.filter(
                    ~pl.col("concept_id").is_in(cdc_to_mult)
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
        multiple_strength = (
            self.strength.data()
            .filter(pl.col("drug_concept_id").is_in(cdc_frame["concept_id"]))
            .select("drug_concept_id", "ingredient_concept_id")
            .group_by("drug_concept_id")
            .count()
            .filter(pl.col("count") > 1)
        )["drug_concept_id"]
        if len(multiple_strength):
            cdc_frame = cdc_frame.filter(
                ~pl.col("concept_id").is_in(multiple_strength)
            )
        self.filter_out_bad_concepts(
            multiple_strength,
            "No Clinical Drug Components with multiple strength entries found",
            "CDC_Multiple_Strength",
            f"Found {len(multiple_strength):,} Clinical Drug Components "
            "with multiple strength entries",
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

            assert not isinstance(strength, dc.LiquidQuantity)

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
            cdc_nodes[concept_id] = node_idx

        self.filter_out_bad_concepts(
            pl.Series(cdc_ingredient_mismatch, dtype=pl.UInt32),
            "All Clinical Drug Components have the same ingredient in "
            "DRUG_STRENGTH and CONCEPT_RELATIONSHIP",
            "CDC_Ing_Mismatch",
            f"{len(cdc_ingredient_mismatch):,} Clinical Drug Components "
            f"had ingredient mismatches between DRUG_STRENGTH and "
            "CONCEPT_RELATIONSHIP",
        )

        self.filter_out_bad_concepts(
            pl.Series(cdc_bad_ingredient, dtype=pl.UInt32),
            "All Clinical Drug Components have valid Ingredients",
            "CDC_Bad_Ing",
            f"{len(cdc_bad_ingredient):,} Clinical Drug Components had "
            "bad Ingredients",
        )

        self.filter_out_bad_concepts(
            pl.Series(cdc_bad_precise_ingredient, dtype=pl.UInt32),
            "All Clinical Drug Components have valid Precise Ingredients",
            "CDC_Bad_PI",
            f"{len(cdc_bad_precise_ingredient):,} Clinical Drug Components "
            "had bad Precise Ingredients",
        )

        self.filter_out_bad_concepts(
            pl.Series(cdc_failed, dtype=pl.UInt32),
            "All Clinical Drug Components were successfully created",
            "CDC_Failed",
            f"{len(cdc_failed):,} Clinical Drug Components had failed creation",
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
            self.strength.reader.anti_join(
                drug_strength_noning, on=["drug_concept_id"]
            )
        else:
            msg = "Unused"

        self.filter_out_bad_concepts(
            drug_strength_noning["drug_concept_id"],
            "No drug concepts that treat non-Ingredients as Ingredients "
            "in DRUG_STRENGTH found",
            "Non_Ing_Strength",
            msg,
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

        self.filter_out_bad_concepts(
            bad_drugs,
            "All units in DRUG_STRENGTH are valid",
            "Deprecated_Unit",
            f"Found {len(bad_drugs):,} drug concepts with deprecated "
            "units in DRUG_STRENGTH",
        )

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
        invalid_drugs = invalid_strength["drug_concept_id"].unique()
        self.filter_out_bad_concepts(
            invalid_drugs,
            "Malformed_Strength",
            "All strength configurations are validated",
            f"Found {len(invalid_strength):,} invalid strength "
            f"configurations for {len(invalid_drugs):,} drug concepts",
        )

    def process_branded_drug_forms(
        self, cdf_nodes: _TempNodeView
    ) -> _TempNodeView:
        """
        Process Branded Drug Forms and link them to parent Clinical Drug
        Components and Brand Names

        Args:
            cdc_nodes: List of node indices for Clinical Drug Components in the
                hierarchy. Required to speed up the linking process.

        Returns:
            List of node indices for Branded Drug Forms in the hierarchy
        """

        self.logger.info("Processing Branded Drug Forms")
        bdf_nodes: _TempNodeView = {}

        self.logger.info("Finding Brand Names for Branded Drug Forms")
        bdf_to_bn = self.get_class_relationships(
            class_id_1="Branded Drug Form",
            class_id_2="Brand Name",
            relationship_id="Has brand name",
        ).select("concept_id", brand_concept_id="concept_id_target")

        bdf_concepts = (
            self.concept.data()
            .filter(pl.col("concept_class_id") == "Branded Drug Form")
            .select("concept_id")
            .join(bdf_to_bn, on="concept_id", how="left")
        )

        # Catch empty brand names
        bdf_no_bn = bdf_concepts.filter(pl.col("brand_concept_id").is_null())
        self.filter_out_bad_concepts(
            bdf_no_bn["concept_id"],
            "All Branded Drug Forms have a Brand Name",
            "BDF_no_BN",
            f"{len(bdf_no_bn):,} Branded Drug Forms had no Brand Name",
        )

        # Catch multiple brand names for a single Branded Drug Form
        bdf_mult_bn = (
            bdf_to_bn.group_by("concept_id").count().filter(pl.col("count") > 1)
        )
        self.filter_out_bad_concepts(
            bdf_mult_bn["concept_id"],
            "All Branded Drug Forms had a single Brand Name",
            "BDF_Mult_BN",
            f"{len(bdf_mult_bn):,} Branded Drug Forms had multiple Brand Names",
        )
        if len(bdf_mult_bn):
            bdf_concepts = bdf_concepts.join(
                bdf_mult_bn, on="concept_id", how="anti"
            )

        # Find Clinical Drug Forms for Branded Drug Forms
        bdf_to_cdf = self.get_class_relationships(
            class_id_1="Branded Drug Form",
            class_id_2="Clinical Drug Form",
            relationship_id="Tradename of",
        ).select("concept_id", cdf_concept_id="concept_id_target")
        bdf_concepts = bdf_concepts.join(
            bdf_to_cdf, on="concept_id", how="left"
        )

        # Catch Branded Drug Forms without Clinical Drug Forms
        bdf_no_cdf = bdf_concepts.filter(pl.col("cdf_concept_id").is_null())
        self.filter_out_bad_concepts(
            bdf_no_cdf["concept_id"],
            "All Branded Drug Forms have a Clinical Drug Form",
            "BDF_no_CDF",
            f"{len(bdf_no_cdf):,} Branded Drug Forms had no Clinical Drug Form",
        )
        if len(bdf_no_cdf):
            bdf_concepts = bdf_concepts.filter(
                pl.col("cdf_concept_id").is_not_null()
            )

        # Catch multiple Clinical Drug Forms for a single Branded Drug Form
        bdf_mult_cdf = (
            bdf_to_cdf.group_by("concept_id")
            .count()
            .filter(pl.col("count") > 1)
        )
        self.filter_out_bad_concepts(
            bdf_mult_cdf["concept_id"],
            "All Branded Drug Forms had a single Clinical Drug Form",
            "BDF_Mult_CDF",
            f"{len(bdf_mult_cdf):,} Branded Drug Forms had multiple "
            "Clinical Drug Forms",
        )
        if len(bdf_mult_cdf):
            bdf_concepts = bdf_concepts.join(
                bdf_mult_cdf, on="concept_id", how="anti"
            )

        # We do not need the ingredient data to construct the BDF concept, but
        # we will cross-check it against CDF data to ensure consistency
        bdf_ing_ds = (
            self.strength.data()
            .join(
                other=bdf_concepts,
                left_on="drug_concept_id",
                right_on="concept_id",
            )
            .select(["drug_concept_id", "ingredient_concept_id"])
            .group_by("drug_concept_id")
            .all()
            .select(
                concept_id="drug_concept_id",
                ingredient_concept_ids="ingredient_concept_id",
            )
        )

        # NOTE: We skip internal consistency checks. There is only one way for
        # a BDF to be correct, and that is to have a single CDF with the same
        # data.

        bdf_concepts = bdf_concepts.join(
            other=bdf_ing_ds,
            on="concept_id",
        )

        bdf_bad_cdf: list[int] = []
        bdf_cdf_ing_mismatch: list[int] = []
        bdf_bad_bn: list[int] = []
        bdf_failed: list[int] = []
        for row in bdf_concepts.iter_rows():
            concept_id: int = row[0]
            brand_concept_id: int = row[1]
            cdf_concept_id: int = row[2]
            ingredient_concept_ids: list[int] = row[3]

            try:
                brand_name = self.atoms.brand_name[
                    dc.ConceptId(brand_concept_id)
                ]
            except KeyError:
                self.logger.error(
                    f"Brand Name {brand_concept_id} not found for "
                    f"Branded Drug Form {concept_id}"
                )
                bdf_bad_bn.append(concept_id)
                continue

            if (cdf_node_idx := cdf_nodes.get(cdf_concept_id)) is None:
                self.logger.error(
                    f"Branded Drug Form {concept_id} had no registered "
                    f"Clinical Drug Form {cdf_concept_id}"
                )
                bdf_bad_cdf.append(concept_id)
                continue

            cdf = self.hierarchy.graph[cdf_node_idx]
            if not isinstance(cdf, dc.ClinicalDrugForm):
                # This should never happen, but we will catch it anyway
                self.logger.error(
                    f"Branded Drug Form {concept_id} specified a non-CDF "
                    f"{cdf_concept_id} as Clinical Drug Form"
                )
                bdf_bad_cdf.append(concept_id)
                continue

            # Ingredients are sorted by identifier, so we can compare them
            cdf_ing_ids = [ing.identifier for ing in cdf.ingredients]

            if sorted(ingredient_concept_ids) != cdf_ing_ids:
                self.logger.error(
                    f"Ingredients mismatch for Branded Drug Form {concept_id}: "
                    f"{ingredient_concept_ids} != {cdf_ing_ids}"
                )
                bdf_cdf_ing_mismatch.append(concept_id)
                continue

            try:
                bdf = dc.BrandedDrugForm(
                    identifier=dc.ConceptId(concept_id),
                    brand_name=brand_name,
                    clinical_drug_form=cdf,
                )
            except RxConceptCreationError as e:
                self.logger.error(
                    f"Failed to create Branded Drug Form {concept_id}: {e}"
                )
                bdf_failed.append(concept_id)
                continue

            node_idx = self.hierarchy.add_branded_drug_form(bdf, cdf_node_idx)
            bdf_nodes[concept_id] = node_idx

        self.filter_out_bad_concepts(
            pl.Series(bdf_bad_bn, dtype=pl.UInt32),
            "All Branded Drug Forms have valid Brand Names",
            "BDF_Bad_BN",
            f"{len(bdf_bad_bn):,} Branded Drug Forms had bad Brand Names",
        )

        self.filter_out_bad_concepts(
            pl.Series(bdf_bad_cdf, dtype=pl.UInt32),
            "All Branded Drug Forms have valid Clinical Drug Forms",
            "BDF_Bad_CDF",
            f"{len(bdf_bad_cdf):,} Branded Drug Forms had bad Clinical Drug "
            f"Forms",
        )

        self.filter_out_bad_concepts(
            pl.Series(bdf_cdf_ing_mismatch, dtype=pl.UInt32),
            "All Branded Drug Forms have matching Ingredients with their "
            "Clinical Drug Forms",
            "BDF_Ing_Mismatch",
            f"{len(bdf_cdf_ing_mismatch):,} Branded Drug Forms had mismatched "
            "Ingredients with their Clinical Drug Forms",
        )

        self.filter_out_bad_concepts(
            pl.Series(bdf_failed, dtype=pl.UInt32),
            "All Branded Drug Forms were successfully created",
            "BDF_Failed",
            f"{len(bdf_failed):,} Branded Drug Forms had failed creation",
        )

        return bdf_nodes
