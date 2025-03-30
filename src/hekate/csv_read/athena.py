"""
Contains implementations to read CSV data from Athena OMOP CDM Vocabularies
"""

from itertools import zip_longest, chain  # For iteration
import logging  # for typing
from abc import ABC  # For shared table reading methods
from collections.abc import Sequence, Mapping  # for typing
from pathlib import Path  # To locate the CSV files
from typing import Literal, NamedTuple, override, overload  # For typing

import polars as pl  # For type hinting and schema definition
from csv_read.generic import CSVReader, Schema  # To read files
from rx_model import drug_classes as dc  # For concept classes
from rx_model import hierarchy as h  # For hierarchy building
from utils.exceptions import (
    InvalidConceptIdError,
    RxConceptCreationError,
)  # For error handling

from rx_model.hierarchy.hosts import NodeIndex
from rx_model import descriptive as d
from utils.classes import SortedTuple, PlRealNumber, PyRealNumber
from utils.constants import (
    ALL_CONCEPT_RELATIONSHIP_IDS,
    ATHENA_OVERFILTERING_TRESHOLD,
    ATHENA_OVERFILTERING_WARNING,
    PERCENT_CONCEPT_ID,
)
from utils.logger import LOGGER
from utils.enums import Cardinality, ConceptClassId as CCId


# Type hint for a dictionary linking int concept_ids to indices in the hierarchy
# graph. This serves as a temporary cache to speed up the hierarchy building
type _TempNodeView = dict[int, NodeIndex]

type _BoxSizeDict = dict[int, int]

type _MonoAttribute = (
    dc.DoseForm[dc.ConceptId]
    | dc.Supplier[dc.ConceptId]
    | dc.BrandName[dc.ConceptId]
)

type _ParentNode = dc.DrugNode[dc.ConceptId, dc.Strength | None]


class _StrengthTuple(NamedTuple):
    ingredient_concept_id: int
    strength: dc.Strength


_get_def = d.ComplexDrugNodeDefinition.get


class OMOPTable[IdS: pl.Series | None](CSVReader[IdS], ABC):
    """
    Abstract class to read Athena OMOP CDM Vocabularies

    Attributes:
        TABLE_SCHEMA: Schema for the table.
        TABLE_COLUMNS: Ordered sequence of columns to keep from the table.
    """

    TABLE_SCHEMA: Schema
    TABLE_COLUMNS: list[str]


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
        self, frame: pl.LazyFrame, valid_concepts: None = None
    ) -> pl.LazyFrame:
        del valid_concepts
        return frame.filter(
            # Invalid reason is needed for filtering, so we keep it
            # pl.col("invalid_reason").is_null(),
            ((pl.col("domain_id") == "Drug") | (pl.col("domain_id") == "Unit")),
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
                | (pl.col("concept_class_id") == "Clinical Drug Box")
                | (pl.col("concept_class_id") == "Branded Drug Box")
                | (pl.col("concept_class_id") == "Quant Clinical Box")
                | (pl.col("concept_class_id") == "Quant Branded Box")
                # | (pl.col("concept_class_id") == "Clinical Pack")
                # | (pl.col("concept_class_id") == "Branded Pack")
                # | (pl.col("concept_class_id") == "Clinical Pack Box")
                # | (pl.col("concept_class_id") == "Branded Pack Box")
            ),
        )

    def get_metadata(self, ids: Sequence[dc.ConceptId]):
        """
        Get metadata for a sequence of concept_ids
        """
        return (
            self.collect()
            .filter(pl.col("concept_id").is_in(ids))
            .select("concept_id", "vocabulary_id", "valid_start_date")
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
        self, frame: pl.LazyFrame, valid_concepts: pl.Series | None = None
    ) -> pl.LazyFrame:
        concept = valid_concepts
        if concept is None:
            raise ValueError("Concept filter argument is required.")

        return frame.filter(
            # Only care about valid internal relationships
            pl.col("invalid_reason").is_null(),
            pl.col("relationship_id").is_in(ALL_CONCEPT_RELATIONSHIP_IDS),
            pl.col("concept_id_1").is_in(concept),
            pl.col("concept_id_2").is_in(concept),
        )


class StrengthTable(OMOPTable[pl.Series]):
    TABLE_SCHEMA: Schema = {
        "drug_concept_id": pl.UInt32,
        "ingredient_concept_id": pl.UInt32,
        "amount_value": PlRealNumber,
        "amount_unit_concept_id": pl.UInt32,
        "numerator_value": PlRealNumber,
        "numerator_unit_concept_id": pl.UInt32,
        "denominator_value": PlRealNumber,
        "denominator_unit_concept_id": pl.UInt32,
        "box_size": pl.UInt16,
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
        "box_size",
    ]

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: pl.Series | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError("Valid drugs as filter argument is required.")

        return frame.filter(
            pl.col("invalid_reason").is_null(),
            pl.col("drug_concept_id").is_in(valid_concepts),
        ).select(pl.all())


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
        self, frame: pl.LazyFrame, valid_concepts: pl.Series | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError("Concept filter argument is required.")

        # Only keep internal relationships
        return frame.filter(
            pl.col("descendant_concept_id").is_in(valid_concepts),
            pl.col("ancestor_concept_id").is_in(valid_concepts),
        )


class OMOPVocabulariesV5:
    """
    Class to read Athena OMOP CDM Vocabularies
    """

    class _StrengthDataRow(NamedTuple):
        """
        Shape of the row data for the output of get_strength_data()
        """

        drug_concept_id: int
        ingredient_concept_id: int
        amount_value: PyRealNumber
        amount_unit_concept_id: int
        numerator_value: PyRealNumber
        numerator_unit_concept_id: int
        denominator_value: PyRealNumber
        denominator_unit_concept_id: int
        box_size: int

        # Boolean flags for strength data shape
        amount_only: bool
        liquid_concentration: bool
        liquid_quantity: bool
        gas_concentration: bool

    def get_class_relationships(
        self, class_id_1: str, class_id_2: str, relationship_id: str
    ) -> pl.DataFrame:
        """
        Get relationships of a defined type between concepts of two classes.
        """
        concept = self.concept.collect()
        relationship = self.relationship.collect().filter(
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

    def get_validated_relationships_view(
        self,
        source_class: CCId,
        relationships: Sequence[d.RelationshipDescription],
        include_name: bool = False,
    ) -> pl.DataFrame:
        """
        Get a DataFrame view of the relationships between a source class and
        target classes, with additional validation checks.
        """

        self.logger.info(f"Finding relationships for {source_class.value}")

        source_abbr = "".join(char[0] for char in source_class.value.split())
        source_concepts = (
            self.concept.collect()
            .filter(pl.col("concept_class_id") == source_class.value)
            .select(
                ["concept_id", "concept_name"] if include_name else "concept_id"
            )
        )

        for rel in relationships:
            target_abbr = rel.target_definition.get_abbreviation()
            target_colname = rel.target_definition.get_colname()
            target_cid = rel.target_definition.class_id

            self.logger.info(f"Finding {target_cid} for {source_class.value}")
            source_to_target = (
                self.get_class_relationships(
                    class_id_1=source_class.value,
                    class_id_2=target_cid,
                    relationship_id=rel.relationship_id,
                )
                .select("concept_id", "concept_id_target")
                .rename({"concept_id_target": target_colname})
            )

            if len(source_to_target) == 0:
                # Programming error
                raise ValueError(
                    f"No relationships of {rel.relationship_id} found for "
                    f"{source_class.value} to {target_cid}"
                )

            # Catch empty attributes
            if rel.cardinality in d.CARDINALITY_REQUIRED:
                source_no_target = source_concepts.join(
                    source_to_target, on="concept_id", how="anti"
                )
                self.filter_out_bad_concepts(
                    len(source_to_target["concept_id"].unique()),
                    source_no_target["concept_id"],
                    f"All {source_class} have a {target_cid}",
                    f"{source_abbr}_no_{target_abbr}",
                    f"{len(source_no_target):,} {source_class} had no "
                    f"{target_cid} relations",
                )
                if len(source_no_target):
                    source_concepts = source_concepts.join(
                        source_no_target, on="concept_id", how="anti"
                    )

            if rel.cardinality in d.CARDINALITY_SINGLE:
                # Catch multiple attributes
                source_mult_target = (
                    source_to_target["concept_id"]
                    .value_counts()
                    .filter(pl.col("count") > 1)
                )

                self.filter_out_bad_concepts(
                    len(source_to_target["concept_id"].unique()),
                    source_mult_target["concept_id"],
                    f"All {source_class} had a single {target_cid}",
                    f"{source_abbr}_Mult_{target_abbr}",
                    f"{len(source_mult_target):,} {source_class} had multiple "
                    f"{target_cid} relations",
                )
                if len(source_mult_target):
                    source_concepts = source_concepts.join(
                        source_mult_target, on="concept_id", how="anti"
                    )

                # Attach the attribute
                source_concepts = source_concepts.join(
                    other=source_to_target,
                    on="concept_id",
                    # NOTE: Empty targets must be caught earlier by cardinality
                    how="left",
                )
            else:  # expected_cardinality in __CARDINALITY_MULTIPLE
                # Attach grouped attributes
                source_concepts = source_concepts.join(
                    other=(
                        source_to_target.group_by("concept_id")
                        .all()
                        .rename({target_colname: target_colname + "s"})
                    ),
                    on="concept_id",
                    how="left",
                )

        return source_concepts

    def get_ds_ingredient_data(
        self, drug_ids: pl.Series, expect_cardinality: Cardinality
    ) -> dict[int, list[int]]:
        """
        Get the ingredient data slice for each drug concept per the
        DRUG_STRENGTH table.

        Args:
            drug_ids: Series of drug concept_ids to filter the data.
            expect_cardinality: Expected cardinality of the relationship, as
                defined in the _Cardinality enum. Note that only ONE and NONZERO
                are valid here. Return type is always a dictionary of lists,
                regardless of the cardinality.

        Returns:
            A dictionary with drug concept_ids as keys and lists of
            correpsonding ingredient concept_ids as values.
        """

        if expect_cardinality not in d.CARDINALITY_REQUIRED:
            raise ValueError(
                "Expected cardinality must be ONE or NONZERO for this method"
            )

        drug_ids = drug_ids.unique()

        ing_data: dict[int, list[int]] = {}
        ing_df = self.strength.collect().join(
            other=drug_ids.to_frame(name="drug_concept_id"),
            on="drug_concept_id",
            how="left",
        )

        # Filter out drugs with no ingredients
        no_ingredients = ing_df.filter(
            pl.col("ingredient_concept_id").is_null()
        )
        self.filter_out_bad_concepts(
            len(drug_ids),
            no_ingredients["drug_concept_id"],
            "All drugs have ingredients",
            "DS_No_Ing",
            f"{len(no_ingredients):,} drugs had no ingredients",
        )
        if len(no_ingredients):
            ing_df = ing_df.filter(
                pl.col("ingredient_concept_id").is_not_null()
            )

        if expect_cardinality is Cardinality.ONE:
            # Filter out drugs with multiple ingredients
            multiple_ingredients = (
                ing_df["drug_concept_id"]
                .value_counts()
                .filter(pl.col("count") > 1)
            )
            self.filter_out_bad_concepts(
                len(drug_ids),
                multiple_ingredients["drug_concept_id"],
                "All drugs have a single ingredient",
                "DS_Mult_Ing",
                f"{len(multiple_ingredients):,} drugs had multiple ingredients",
            )
            if len(multiple_ingredients):
                ing_df = ing_df.join(
                    multiple_ingredients, on="drug_concept_id", how="anti"
                )

        for row in ing_df.iter_rows():
            drug_id: int = row[0]
            ingredient_id: int = row[1]
            if (existing := ing_data.get(drug_id)) is None:
                ing_data[drug_id] = [ingredient_id]
            else:
                existing.append(ingredient_id)

        return ing_data

    @overload
    def get_strength_data(
        self,
        drug_ids: pl.Series,
        expect_cardinality: Cardinality,
        accepted_configurations: tuple[d.StrengthConfiguration, ...],
        expect_box_size: Literal[False],
    ) -> tuple[dict[int, list[_StrengthTuple]], None]: ...

    @overload
    def get_strength_data(
        self,
        drug_ids: pl.Series,
        expect_cardinality: Cardinality,
        accepted_configurations: tuple[d.StrengthConfiguration, ...],
        expect_box_size: Literal[True],
    ) -> tuple[dict[int, list[_StrengthTuple]], _BoxSizeDict]: ...

    def get_strength_data(
        self,
        drug_ids: pl.Series,
        expect_cardinality: Cardinality,
        accepted_configurations: tuple[d.StrengthConfiguration, ...],
        expect_box_size: bool,
    ) -> tuple[dict[int, list[_StrengthTuple]], _BoxSizeDict | None]:
        """
        Get the strength data slice for each drug concept.

        This does not validate the adherence of the concept to a particular
        strength data shape. This should be done either as a follow-up
        integrity check or in a separate method.

        The concept will be rejected only if it fails to match any of the
        valid configurations, or if it's entries match multiple configurations.

        Args:
            drug_ids: Series of drug concept_ids to filter the data.
            expect_cardinality: Expected cardinality of the relationship, as
                defined in the Cardinality enum. Note that only ONE and NONZERO
                are valid here. Return type is always a dictionary of lists,
                regardless of the cardinality.
            accepted_configurations: Iterable of strength data classes that are
                accepted for the provided drug_ids.
            expect_box_size: Boolean flag indicating whether to expect the box
                size for the concepts. Missing this expectation will lead to
                exclusion of those concepts.

        Returns:
            A tuple with two elements:
             * A dictionary with drug concept_ids as keys and strength entries
             as values. A strength entry is a tuple with an integer concept_id
             and a variant of strength data, in shape of `SolidStrength`,
             `LiquidQuantity`, `GasPercentage` or `LiquidConcentration`.
             * Either None or an integer box size, if the box size is expected.
        """
        if expect_cardinality not in d.CARDINALITY_REQUIRED:
            raise ValueError(
                "Expected cardinality must be ONE or NONZERO for this method"
            )

        if len(accepted_configurations) == 0:
            raise ValueError(
                "At least one strength configuration must be specified"
            )
        accepted_configurations_cls = tuple(
            cfg.value for cfg in accepted_configurations
        )

        concepts = drug_ids.unique().to_frame(name="drug_concept_id")

        strength_data: dict[int, list[_StrengthTuple]] = {}
        strength_df = concepts.join(
            other=self.strength.collect(),
            on="drug_concept_id",
            how="left",
        )

        # Check and process box size
        mult_box_size = (
            strength_df.select("drug_concept_id", "box_size")
            .group_by("drug_concept_id")
            .n_unique()
            .filter(pl.col("box_size") > 1)
        )
        self.filter_out_bad_concepts(
            len(drug_ids),
            mult_box_size["drug_concept_id"],
            "All drugs have a single box size",
            "DS_Mult_Box",
            f"{len(mult_box_size):,} drugs had multiple box sizes",
        )
        if len(mult_box_size):
            strength_df = strength_df.join(
                mult_box_size, on="drug_concept_id", how="anti"
            )

        box_size: dict[int, int] | None
        if expect_box_size:
            # Filter out drugs with no box size
            no_box_size = strength_df.filter(pl.col("box_size").is_null())
            self.filter_out_bad_concepts(
                len(drug_ids),
                no_box_size["drug_concept_id"],
                "All drugs have a box size",
                "DS_No_Box",
                f"{len(no_box_size):,} drugs had no box sizes",
            )
            if len(no_box_size):
                strength_df = strength_df.filter(
                    pl.col("box_size").is_not_null()
                )

            box_size = dict(
                strength_df.select("drug_concept_id", "box_size")
                .unique()
                .iter_rows()
            )

        else:
            # Make sure all box sizes are None
            has_box_size = strength_df.filter(pl.col("box_size").is_not_null())
            self.filter_out_bad_concepts(
                len(drug_ids),
                has_box_size["drug_concept_id"],
                "No drugs specify box sizes",
                "DS_Redun_Box",
                f"{len(has_box_size):,} drugs had redundant box sizes",
            )
            if len(has_box_size):
                strength_df = strength_df.filter(pl.col("box_size").is_null())
            box_size = None

        # Filter out drugs with no strength data
        # NOTE: Due to design of OMOP, this is highly unlikely
        no_strength = strength_df.filter(
            pl.col("ingredient_concept_id").is_null()
        )
        self.filter_out_bad_concepts(
            len(drug_ids),
            no_strength["drug_concept_id"],
            "All drugs have strength data",
            "DS_No_Strength",
            f"{len(no_strength):,} drugs had no strength data",
        )
        if len(no_strength):
            strength_df = strength_df.filter(
                pl.col("ingredient_concept_id").is_not_null()
            )

        if expect_cardinality is Cardinality.ONE:
            # Filter out drugs with multiple strength data
            multiple_strength = (
                strength_df["drug_concept_id"]
                .value_counts()
                .filter(pl.col("count") > 1)
            )
            self.filter_out_bad_concepts(
                len(drug_ids),
                multiple_strength["drug_concept_id"],
                "All drugs have a single strength data entry",
                "DS_Mult_Strength",
                f"{len(multiple_strength):,} drugs had multiple strength data "
                "entries",
            )
            if len(multiple_strength):
                strength_df = strength_df.join(
                    multiple_strength, on="drug_concept_id", how="anti"
                )

        # We want to be very strict about the strength data, so we will
        # look for explicit data shapes
        for cfg, expression in d.STRENGTH_CONFIGURATIONS_ID.items():
            strength_df = strength_df.with_columns(**{cfg.name: expression})
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
            row = self._StrengthDataRow._make(row)

            if row.drug_concept_id in failed_concept_ids:
                # We already know this concept is bad
                continue

            # Pick a Strength variant based on the determined configuration
            strength: dc.Strength
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
                        strength = dc.GasPercentage(
                            numerator_value=row.numerator_value,
                            numerator_unit=get_unit(
                                row.numerator_unit_concept_id
                            ),
                        )
                    case _:
                        self.logger.debug(
                            f"Strength data for {row.drug_concept_id} had "
                            f"unexpected value configuration for "
                            f"{row.ingredient_concept_id}"
                        )
                        failed_concept_ids.append(row.drug_concept_id)
                        continue
            except RxConceptCreationError as e:
                self.logger.debug(
                    f"Failed to create strength data for {row.drug_concept_id}"
                    f": {e}"
                )
                failed_concept_ids.append(row.drug_concept_id)
                continue

            if not isinstance(strength, accepted_configurations_cls):  # pyright: ignore[reportUnnecessaryIsInstance]  # noqa: E501
                self.logger.debug(
                    f"Strength data {strength} for {row.drug_concept_id} did "
                    f"not match any of the accepted configurations of its class"
                )
                failed_concept_ids.append(row.drug_concept_id)

            if (existing := strength_data.get(row.drug_concept_id)) is None:
                strength_data[row.drug_concept_id] = [
                    _StrengthTuple(row.ingredient_concept_id, strength)
                ]
            else:
                # Assert that the configuration matches the other rows
                if not isinstance(strength, type(existing[0].strength)):
                    self.logger.debug(
                        f"Strength data for {row.drug_concept_id} had "
                        f"inconsistent configurations between "
                        f"{row.ingredient_concept_id} "
                        f"and {existing[0].ingredient_concept_id}"
                    )
                    failed_concept_ids.append(row.drug_concept_id)
                    continue

                # Assert that the denominator values match for LiquidQuantity
                if isinstance(strength, dc.LiquidQuantity):
                    assert isinstance(existing[0].strength, dc.LiquidQuantity)
                    if not strength.denominator_matches(existing[0].strength):
                        self.logger.debug(
                            f"Strength data for {row.drug_concept_id} had "
                            f"inconsistent denominator values between "
                            f"{row.ingredient_concept_id} "
                            f"and {existing[0].ingredient_concept_id}"
                        )
                        failed_concept_ids.append(row.drug_concept_id)
                        continue

                # Put the new strength data in the list
                strength_data[row.drug_concept_id].append(
                    _StrengthTuple(row.ingredient_concept_id, strength)
                )

            # TODO: save strength data to self.strengths
            # self.strengths.add_strength(row.ingredient_concept_id, strength)

        self.filter_out_bad_concepts(
            len(drug_ids),
            pl.Series(failed_concept_ids, dtype=pl.UInt32),
            "All strength data was successfully created",
            "Strength_Creation",
            f"{len(failed_concept_ids):,} drug concepts had failed "
            "strength data creation",
        )

        return strength_data, box_size

    def filter_strength_chunk(self, strength_chunk: pl.DataFrame) -> pl.Series:
        """
        Apply integrity checks to a strength chunk. Intended to be only called
        from `filter_invalid_strength_configurations` method. This method will
        return a pl.Series of only passing concept_ids.
        """

        # Find drugs that match no known configuration
        # NOTE: This is probably redundant after initial validation
        invalid_mask = (
            strength_chunk.select([
                s.name for s in d.StrengthConfiguration
            ]).sum_horizontal()
        ) == 0

        if n_invalid := invalid_mask.sum():
            self.logger.debug(
                f"{n_invalid} drug concepts have invalid strength data and "
                f"will be excluded from the processing"
            )
            strength_chunk = strength_chunk.filter(~invalid_mask)
        else:
            self.logger.info("No invalid strength configurations found")

        # Find drugs that match more than one configuration over rows
        collapsed_df = (
            strength_chunk.select(
                "drug_concept_id",
                *[s.name for s in d.StrengthConfiguration],
            )
            .group_by("drug_concept_id")
            .max()  # T > F
        )
        muliple_match_mask = (
            collapsed_df.select([
                s.name for s in d.StrengthConfiguration
            ]).sum_horizontal()
            > 1
        )

        if n_unmatched := muliple_match_mask.sum():
            self.logger.error(
                f"{n_unmatched} drug concepts have ambiguously structured "
                f"strength data and will be excluded from the processing"
            )
            multiple_match_ids = collapsed_df["drug_concept_id"].filter(
                muliple_match_mask
            )
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
            self.logger.debug(
                f"{n_ambiguous_denom} drug concepts have ambiguous "
                f"denominator values and will be excluded from the processing"
            )
            ambiguous_denom_ids = drugs_with_denom_counts[
                "drug_concept_id"
            ].filter(ambiguous_denom_mask)
            strength_chunk = strength_chunk.filter(
                ~pl.col("drug_concept_id").is_in(ambiguous_denom_ids)
            )
        else:
            self.logger.info("No ambiguous denominator values found")

        return strength_chunk["drug_concept_id"]

    def __init__(self, vocab_download_path: Path):
        self.logger: logging.Logger = LOGGER.getChild(self.__class__.__name__)

        # Initiate hierarchy containers
        self.atoms: h.Atoms[dc.ConceptId] = h.Atoms(self.logger)
        self.strengths: h.KnownStrengths[dc.ConceptId] = h.KnownStrengths()

        self.hierarchy: h.RxHierarchy[dc.ConceptId] = h.RxHierarchy()
        self.hierarchy.set_logger(self.logger)

        self.logger.info(
            f"Starting processing of Athena Vocabularies from "
            f"{vocab_download_path}"
        )

        # Vocabulary table readers
        self.concept: ConceptTable = ConceptTable(
            path=vocab_download_path / "CONCEPT.csv",
        )

        # Materialize the concept table to filter big tables early
        self.concept.materialize()
        all_concept_ids = self.concept.collect()["concept_id"]

        self.relationship: RelationshipTable = RelationshipTable(
            path=vocab_download_path / "CONCEPT_RELATIONSHIP.csv",
            reference_data=all_concept_ids,
        )

        self.ancestor: AncestorTable = AncestorTable(
            path=vocab_download_path / "CONCEPT_ANCESTOR.csv",
            reference_data=all_concept_ids,
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
            reference_data=self.concept.collect()["concept_id"],
        )

        # Also use Drug Strength for filtering other tables
        self.filter_deprecated_units_in_strength()
        self.filter_non_ingredient_in_strength()
        self.filter_invalid_strength_configurations()
        self.filter_too_high_percentages_in_strength()

        # Process the drug classes from the simplest to the most complex
        self.process_atoms()
        self.process_precise_ingredients()

        cdf_nodes = self.add_class_nodes(
            class_id=CCId.CDF,
            all_parent_nodes={},
            require_parent_match={},
        )
        bdf_nodes = self.add_class_nodes(
            class_id=CCId.BDF,
            all_parent_nodes={CCId.CDF: cdf_nodes},
            require_parent_match={CCId.CDF: [CCId.DOSE_FORM]},
        )
        cdc_nodes = self.add_class_nodes(
            class_id=CCId.CDC,
            all_parent_nodes={},
            require_parent_match={},
        )
        bdc_nodes = self.add_class_nodes(
            class_id=CCId.BDC,
            all_parent_nodes={CCId.CDC: cdc_nodes},
            require_parent_match={},
        )
        cd_nodes = self.add_class_nodes(
            class_id=CCId.CD,
            all_parent_nodes={CCId.CDC: cdc_nodes, CCId.CDF: cdf_nodes},
            require_parent_match={CCId.CDF: [CCId.DOSE_FORM]},
        )
        bd_nodes = self.add_class_nodes(
            class_id=CCId.BD,
            all_parent_nodes={
                CCId.BDC: bdc_nodes,
                CCId.BDF: bdf_nodes,
                CCId.CD: cd_nodes,
            },
            require_parent_match={
                CCId.BDF: [CCId.DOSE_FORM, CCId.BRAND_NAME],
                CCId.BDC: [CCId.BRAND_NAME],
                CCId.CD: [CCId.DOSE_FORM],
            },
        )
        qcd_nodes = self.add_class_nodes(
            class_id=CCId.QCD,
            all_parent_nodes={CCId.CD: cd_nodes},
            require_parent_match={CCId.CD: [CCId.DOSE_FORM]},
        )
        qbd_nodes = self.add_class_nodes(
            class_id=CCId.QBD,
            all_parent_nodes={CCId.BD: bd_nodes, CCId.QCD: qcd_nodes},
            require_parent_match={
                CCId.BD: [CCId.BRAND_NAME, CCId.DOSE_FORM],
                CCId.QCD: [CCId.DOSE_FORM],
            },
        )

        cdb_nodes: _TempNodeView = self.add_class_nodes(
            class_id=CCId.CDB,
            all_parent_nodes={CCId.CD: cd_nodes},
            require_parent_match={CCId.CD: [CCId.DOSE_FORM]},
        )

        # TODO: process the remaining classes
        del qbd_nodes, cdb_nodes

    def process_atoms(self) -> None:
        """
        Process atom concepts with known concept data.
        """
        # Populate atoms with known concepts
        self.logger.info(
            "Processing atomic concepts (Ingredient, Dose Form, etc.)"
        )
        rxn_atoms: pl.DataFrame = (
            self.concept.collect()
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
        ing_to_precise = self.get_validated_relationships_view(
            source_class=CCId.PRECISE_INGREDIENT,
            relationships=[
                d.RelationshipDescription(
                    relationship_id="Form of",  # Maps to?
                    cardinality=Cardinality.ONE,
                    target_definition=d.INGREDIENT_DEFINITION,
                ),
            ],
            include_name=True,
        )

        self.logger.info("Processing Precise Ingredients")
        column_names = ing_to_precise.columns
        concept_id_target_idx = column_names.index("ingredient_concept_id")
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
            self.concept.collect()
            .filter(
                pl.col("standard_concept") == "S",
                pl.col("concept_class_id") == "Ingredient",
            )
            .join(
                other=self.ancestor.collect(),
                left_on="concept_id",
                right_on="ancestor_concept_id",
            )
            .select(concept_id="descendant_concept_id")
        )

        orphaned_complex_concepts = (
            self.concept.collect()
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
            len(ingredient_descendants) + len(orphaned_complex_concepts),
            orphaned_complex_concepts["concept_id"],
            "All complex drug concepts have a valid Ingredient ancestor",
            "Orphaned_Complex",
            f"{len(orphaned_complex_concepts):,} complex drug concepts have "
            f"no ancestor Ingredient",
        )

        # NOTE: We should also check for broken hierarchy links in between the
        # complex drug concepts themselves, but it is not as important for
        # practical applications. If such breakages exist, they will be caught
        # by the lack of path to ingredients.

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
            self.concept.collect()
            .filter(
                pl.col("vocabulary_id") == "RxNorm",
                pl.col("concept_class_id") == "Precise Ingredient",
                pl.col("invalid_reason").is_null(),
            )
            .select(["concept_id"])
        )
        complex_drug_concepts = self.concept.collect().filter(
            pl.col("standard_concept") == "S",
            ~(pl.col("concept_class_id") == "Ingredient"),
        )
        complex_pi_as_ing = (
            prec_ing_concepts.join(
                other=self.relationship.collect().filter(
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
            len(complex_drug_concepts),
            complex_pi_as_ing["concept_id"],
            "No drug concepts that treat Precise Ingredients as "
            "Ingredients found",
            "PI_as_Ing",
            f"Found {len(complex_pi_as_ing):,} drug concepts that treat "
            f"Precise Ingredients as Ingredients",
        )

    def filter_out_bad_concepts(
        self,
        total_count: int,
        bad_concepts: pl.Series,
        message_ok: str,
        reason_short: str,
        reason_full: str,
    ) -> None:
        """
        Filter out concepts and their relationships from the tables.

        Args:
            total_count: Total number of concepts out of which bad concepts
                are a subset. Used to warn about overfiltering.
            bad_concepts: Polars Series with `concept_id` values to filter out.
            message_ok: Message to log if no bad concepts are found.
            reason_short: Short reason for filtering out the concepts. Will be
                used for structuring the log messages and reports.
            reason_full: Reason for filtering out the concepts. Will be used for
                logging and/or reporting.
        """
        logger = self.logger.getChild(reason_short)

        bad_concepts = bad_concepts.unique()

        if not len(bad_concepts):
            logger.info(message_ok)
            return

        logger.warning(reason_full)
        bad_concepts_df = bad_concepts.to_frame(name="concept_id")

        if ATHENA_OVERFILTERING_WARNING and total_count > 0:
            if len(bad_concepts) / total_count > ATHENA_OVERFILTERING_TRESHOLD:
                logger.warning(
                    f"Overfiltering detected: {len(bad_concepts):,} concepts "
                    f"out of {total_count:,} will be removed. Reason: "
                    f"{reason_full}"
                )
                print(bad_concepts_df)
                answer = input("Continue? [y/N] ")
                if answer.lower() != "y":
                    logger.warning("Operation aborted")
                    raise ValueError(reason_full)

        logger.info("Including all descendants of bad concepts")
        bad_descendants_df = (
            self.ancestor.collect()
            .join(
                other=bad_concepts_df,
                left_on="ancestor_concept_id",
                right_on="concept_id",
            )
            .select(concept_id="descendant_concept_id")
            .unique()
        )
        bad_concepts_df = pl.concat([bad_concepts_df, bad_descendants_df])

        logger.debug("Removing from the concept table")
        self.concept.anti_join(bad_concepts_df, on=["concept_id"])

        logger.debug("Removing from relationship table (left)")
        self.relationship.anti_join(
            bad_concepts_df,
            left_on=["concept_id_1"],
            right_on=["concept_id"],
        )
        logger.debug("Removing from relationship table (right)")
        self.relationship.anti_join(
            bad_concepts_df,
            left_on=["concept_id_2"],
            right_on=["concept_id"],
        )

        logger.debug("Removing from ancestor table (left)")
        self.ancestor.anti_join(
            bad_concepts_df,
            left_on=["ancestor_concept_id"],
            right_on=["concept_id"],
        )

        logger.debug("Removing from ancestor table (right)")
        self.ancestor.anti_join(
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
        strength.anti_join(
            bad_concepts_df,
            left_on=["drug_concept_id"],
            right_on=["concept_id"],
        )
        logger.info("Removing from strength table (right)")
        strength.anti_join(
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
            self.concept.collect()
            .filter(pl.col("invalid_reason").is_not_null())
            .select("concept_id")
        )
        self.logger.info(
            f"Found {len(invalid_concepts):,} invalid concepts "
            f"out of {len(self.concept.collect()):,}"
        )

        # NOTE: Do not use filter_out_bad_concepts() here, this is faster
        self.concept.filter(pl.col("invalid_reason").is_null())
        self.relationship.filter(
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
        complex_drug_concepts = self.concept.collect().filter(
            pl.col("standard_concept") == "S",
            ~(pl.col("concept_class_id") == "Ingredient"),
        )

        for attribute_rel in d.MONO_ATTRIBUTE_DEFINITIONS.values():
            definition = attribute_rel.target_definition
            assert definition is not None
            definition_class_id = definition.omop_concept_class_id.value
            definition_abbv = definition.get_abbreviation()
            self.logger.info(
                f"Salvaging concepts that specify more than one "
                f"{definition_class_id} attribute"
            )

            # Find concepts with multiple defining attributes
            rel_to_attribute = (
                complex_drug_concepts.join(
                    other=self.relationship.collect().filter(
                        pl.col("relationship_id")
                        == attribute_rel.relationship_id
                    ),
                    left_on="concept_id",
                    right_on="concept_id_1",
                    suffix="_relationship",
                )
                .join(
                    other=self.concept.collect().filter(
                        pl.col("concept_class_id") == definition_class_id
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
                len(complex_drug_concepts),
                concept_to_multiple_valid,
                message_ok="No drug concepts with multiple valid "
                f"{definition_class_id} attributes found",
                reason_short="Multiple_" + definition_abbv,
                reason_full=f"Found {len(concept_to_multiple_valid):,} drug "
                f"concepts with multiple valid "
                f"{definition_class_id} attributes",
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
                len(complex_drug_concepts),
                concept_has_only_invalid,
                f"No drug concepts with only invalid "
                f"{definition_class_id} attributes found",
                "No_" + definition_abbv,
                f"Found {len(concept_has_only_invalid):,} drug concepts "
                f"with only invalid {definition_class_id} attributes",
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
                    f"{definition_class_id} attributes"
                )
            else:
                self.logger.info(
                    f"No drug concepts with multiple "
                    f"{definition_class_id} attributes were "
                    f"salvaged"
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
            self.strength.collect()
            .join(
                other=self.concept.collect(),
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
            self.strength.anti_join(
                drug_strength_noning, on=["drug_concept_id"]
            )
        else:
            msg = "Unused"

        self.filter_out_bad_concepts(
            len(
                self.concept.collect().filter(pl.col("standard_concept") == "S")
            ),
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
        valid_units = self.concept.collect().filter(
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

        bad_drugs = (self.strength.collect().filter(bad_units_expr))[
            "drug_concept_id"
        ].unique()

        self.filter_out_bad_concepts(
            len(
                self.concept.collect().filter(pl.col("standard_concept") == "S")
            ),
            bad_drugs,
            "All units in DRUG_STRENGTH are valid",
            "Deprecated_Unit",
            f"Found {len(bad_drugs):,} drug concepts with deprecated "
            "units in DRUG_STRENGTH",
        )

    def filter_too_high_percentages_in_strength(self):
        """
        Filter out drugs that cumulatively have more than 100% of gas
        percentage in DRUG_STRENGTH
        """
        over_strength_percentage = (
            self.strength.collect()
            .filter(
                pl.col("numerator_unit_concept_id") == PERCENT_CONCEPT_ID,
            )
            .select("drug_concept_id", percentage="numerator_value")
            .group_by("drug_concept_id")
            .sum()
            .filter(pl.col("percentage") > 100.0)
        )["drug_concept_id"]

        self.filter_out_bad_concepts(
            len(
                self.concept.collect().filter(pl.col("standard_concept") == "S")
            ),
            over_strength_percentage,
            "All drugs have less than 100% of gas percentage in DRUG_STRENGTH",
            "Gases_Over_100",
            f"Found {len(over_strength_percentage):,} drug concepts with "
            "more than 100% of gas percentage in DRUG_STRENGTH",
        )

    def filter_invalid_strength_configurations(self):
        """
        Filter out drug concepts with invalid strength configurations
        """
        # First, define the valid configurations
        strength_with_class = (
            self.strength.collect()
            .join(
                other=self.concept.collect(),
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
            # Comps and Drugs can have either amount or concentration
            (
                pl.col("concept_class_id").is_in([
                    "Clinical Drug Comp",
                    "Branded Drug Comp",
                    "Clinical Drug",
                    "Branded Drug",
                    "Clinical Drug Box",
                    "Branded Drug Box",
                ])
                & (
                    d.STRENGTH_CONFIGURATIONS_ID[
                        d.StrengthConfiguration.AMOUNT_ONLY
                    ]
                    | d.STRENGTH_CONFIGURATIONS_ID[
                        d.StrengthConfiguration.LIQUID_CONCENTRATION
                    ]
                    | d.STRENGTH_CONFIGURATIONS_ID[
                        d.StrengthConfiguration.GAS_PERCENTAGE
                    ]
                )
            )
            |
            # Quant Drug classes can have only have liquid quantity
            (
                pl.col("concept_class_id").is_in([
                    "Quant Clinical Drug",
                    "Quant Branded Drug",
                    "Quant Clinical Box",
                    "Quant Branded Box",
                ])
                & d.STRENGTH_CONFIGURATIONS_ID[
                    d.StrengthConfiguration.LIQUID_QUANTITY
                ]
            )
        )

        invalid_strength = strength_with_class.filter(~valid_strength)
        invalid_drugs = invalid_strength["drug_concept_id"].unique()
        self.filter_out_bad_concepts(
            len(
                self.concept.collect().filter(pl.col("standard_concept") == "S")
            ),
            invalid_drugs,
            "All strength configurations are validated",
            "Malformed_Strength",
            f"Found {len(invalid_strength):,} invalid strength "
            f"configurations for {len(invalid_drugs):,} drug concepts",
        )

    def add_class_nodes(
        self,
        class_id: CCId,
        all_parent_nodes: dict[CCId, _TempNodeView],
        require_parent_match: dict[CCId, list[CCId]],
    ) -> _TempNodeView:
        """
        Process a set of class nodes and add them to the hierarchy.
        Args:
            class_id: The concept_class_id of the class to be added
            all_parent_nodes: A dictionary of parent nodes indexed by their
                class_id dictionary values are the node indices of the parent
                nodes in the hierarchy.
            require_parent_match: A dictionary where keys are parent classes
                and values are lists of attribute classes that must match
                between the parent and the predicate.

        Returns:
            A dictionary of the node indices of the class nodes indexed by their
            concept_id.
        """
        # Convert input to definitions args
        definition = d.ComplexDrugNodeDefinition.get(class_id)
        p_def_nodes = {
            d.ComplexDrugNodeDefinition.get(p_id): p_view
            for p_id, p_view in all_parent_nodes.items()
        }
        p_def_a_def: dict[
            d.ComplexDrugNodeDefinition, list[d.MonoAtributeDefiniton]
        ] = {}
        for p_id, a_ids in require_parent_match.items():
            attrs: list[d.MonoAtributeDefiniton] = []
            for a_id in a_ids:
                match a_id:
                    case CCId.DOSE_FORM:
                        attrs.append(d.DOSE_FORM_DEFINITION)
                    case CCId.BRAND_NAME:
                        attrs.append(d.BRAND_NAME_DEFINITION)
                    case CCId.SUPPLIER:
                        attrs.append(d.SUPPLIER_DEFINITION)
                    case _:
                        raise ValueError(f"Unexpected attribute class {a_id}")
            p_def_a_def[d.ComplexDrugNodeDefinition.get(p_id)] = attrs

            p_def_a_def[d.ComplexDrugNodeDefinition.get(p_id)]

        self.logger.info(f"Processing {definition.class_id} nodes")
        out_nodes: _TempNodeView = {}

        # Test that all parent definitions come with nodes -- programming error
        for parent_rel in definition.parent_relations:
            if (p_def := parent_rel.target_definition) not in p_def_nodes:
                assert p_def is not None
                raise ValueError(
                    f"Definition {p_def.class_id} not found in parent_nodes"
                )

        # Test that required attrribute keys match the parents -- ditto
        if any(p_def not in p_def_nodes for p_def in p_def_a_def):
            raise ValueError(
                "Parent definitions in require_parent_match do not subset "
                "all_parent_nodes"
            )

        relationship_definitions = [
            *definition.attribute_definitions,
            *definition.parent_relations,
        ]

        if definition.defines_explicit_ingredients:
            relationship_definitions.append(
                d.RelationshipDescription(
                    relationship_id="RxNorm has ing",
                    cardinality=definition.ingredient_cardinality,
                    target_definition=d.INGREDIENT_DEFINITION,
                )
            )

        if definition.defines_explicit_precise_ingredients:
            # Make required cardinality optional
            pi_cardinality: Cardinality
            match definition.ingredient_cardinality:
                case Cardinality.ONE:
                    pi_cardinality = Cardinality.OPTIONAL
                case Cardinality.NONZERO:
                    # NOTE: Not happens now
                    pi_cardinality = Cardinality.ANY
                case _:
                    # Must be unreachable
                    raise ValueError(
                        "Unexpected ingredient cardinality for "
                        "Precise Ingredient definition"
                    )
            relationship_definitions.append(
                d.RelationshipDescription(
                    relationship_id="Has precise ing",
                    cardinality=pi_cardinality,
                    target_definition=d.PRECISE_INGREDIENT_DEFINITION,
                )
            )

        node_concepts = self.get_validated_relationships_view(
            source_class=definition.omop_concept_class_id,
            relationships=relationship_definitions,
        )

        if len(definition.allowed_strength_configurations) == 0:
            node_box_size = None
            node_strength = None
            node_ingreds = self.get_ds_ingredient_data(
                node_concepts["concept_id"],
                expect_cardinality=definition.ingredient_cardinality,
            )
        else:
            node_ingreds = None
            node_strength, node_box_size = self.get_strength_data(
                node_concepts["concept_id"],
                expect_cardinality=definition.ingredient_cardinality,
                accepted_configurations=definition.allowed_strength_configurations,
                expect_box_size=definition.defines_box_size,
            )

        # Bad relations
        node_bad_attr: dict[d.MonoAtributeDefiniton, list[int]] = {}
        node_bad_parent: dict[d.ComplexDrugNodeDefinition, list[int]] = {}

        # Bad ingredients/strengths
        node_bad_ingred: list[int] = []
        node_bad_pi: list[int] = []
        node_ingred_ds_mismatch: list[int] = []

        # Mismatch with parents on attributes
        node_attr_mismatch: dict[
            d.ComplexDrugNodeDefinition,  # Parent definition
            dict[
                d.MonoAtributeDefiniton,  # Attribute definition
                list[int],  # Concept IDs
            ],
        ] = {}

        # Mismatch with parents on ingredients and strengths
        node_strength_mismatch: dict[
            d.ComplexDrugNodeDefinition,  # Parent definition
            list[int],  # Concept IDs
        ] = {}

        # Fail on creation
        node_failed: list[int] = []

        for row in node_concepts.iter_rows():
            # Consume the row
            listed = iter(row)

            # Own concept_id
            concept_id: int = next(listed)

            # Attribute data
            attr_data: dict[d.MonoAtributeDefiniton, _MonoAttribute] = {}
            for attr_rel in definition.attribute_definitions:
                assert isinstance(
                    attr_rel.target_definition,
                    d.MonoAtributeDefiniton,
                )
                attr_id: int = next(listed)

                # Lookup the atom
                try:
                    atom = self.atoms.lookup_unknown(dc.ConceptId(attr_id))
                except InvalidConceptIdError:
                    self.logger.debug(
                        f"Attribute {attr_id} not found for "
                        f"{definition.class_id} {concept_id}"
                    )
                    node_bad_attr.setdefault(
                        attr_rel.target_definition, []
                    ).append(concept_id)
                    continue

                # Test atom class. Catching this means a programming error
                if not isinstance(atom, attr_rel.target_definition.constructor):
                    raise ValueError(
                        f"Expected {attr_id} to be of class "
                        f"{attr_rel.target_definition.class_id} "
                        f"for {concept_id}, got {type(atom)}"
                    )
                attr_data[attr_rel.target_definition] = atom  # pyright: ignore[reportArgumentType]

            # Parent concepts
            parent_indices: list[NodeIndex] = []
            parent_data: dict[
                d.ComplexDrugNodeDefinition, list[_ParentNode]
            ] = {}
            for parent_rel in definition.parent_relations:
                parent_def = parent_rel.target_definition
                assert isinstance(
                    parent_def,
                    d.ComplexDrugNodeDefinition,
                )
                parent_id_or_ids: int | list[int] = next(listed)

                # For simplicity, convert to iterable
                parent_ids: list[int]
                if isinstance(parent_id_or_ids, int):
                    if not parent_rel.cardinality == Cardinality.ONE:
                        # Programming error
                        raise ValueError(
                            f"Expected single parent ID for "
                            f"{parent_def.class_id}, "
                            f"got {parent_id_or_ids}"
                        )
                    parent_ids = [parent_id_or_ids]
                else:
                    parent_ids = parent_id_or_ids

                # Lookup the nodes
                parent_node_view = p_def_nodes[parent_def]
                for parent_id in parent_ids:
                    try:
                        parent_node_idx = parent_node_view[parent_id]
                    except KeyError:
                        self.logger.debug(
                            f"Parent {parent_id} not found for "
                            f"{definition.class_id} {concept_id}"
                        )
                        node_bad_parent.setdefault(parent_def, []).append(
                            concept_id
                        )
                        continue

                    # Try getting the node
                    try:
                        parent_node = self.hierarchy[parent_node_idx]
                    except IndexError:
                        self.logger.debug(
                            f"Parent {parent_id} not found in hierarchy for "
                            f"{definition.class_id} {concept_id}"
                        )
                        node_bad_parent.setdefault(parent_def, []).append(
                            concept_id
                        )
                        continue

                    # Check class (programming error)
                    if not isinstance(parent_node, parent_def.constructor):  # pyright: ignore[reportUnknownMemberType]  # noqa: E501
                        raise ValueError(
                            f"Expected {parent_id} to be of class "
                            f"{parent_def.class_id} for {concept_id}, "
                            f"got {type(parent_node)}"
                        )

                    parent_data.setdefault(parent_def, []).append(parent_node)
                    parent_indices.append(parent_node_idx)

            explicit_ingredients: list[dc.Ingredient[dc.ConceptId]] = []
            if definition.defines_explicit_ingredients:
                ingred_id_or_ids: int | list[int] = next(listed)

                # For simplicity, convert to iterable
                ingred_ids: list[int]
                if isinstance(ingred_id_or_ids, int):
                    if not definition.ingredient_cardinality == Cardinality.ONE:
                        # Programming error
                        raise ValueError(
                            f"Expected single ingredient ID for "
                            f"{definition.class_id}, got {ingred_id_or_ids}"
                        )
                    ingred_ids = [ingred_id_or_ids]
                else:
                    ingred_ids = ingred_id_or_ids

                # Lookup the atoms
                for ingred_id in ingred_ids:
                    try:
                        ingred = self.atoms.ingredient[dc.ConceptId(ingred_id)]
                    except KeyError:
                        self.logger.debug(
                            f"Ingredient {ingred_id} not found for "
                            f"{definition.class_id} {concept_id}"
                        )
                        node_bad_ingred.append(concept_id)
                        continue
                    explicit_ingredients.append(ingred)

            explicit_pis: list[dc.PreciseIngredient] = []
            if definition.defines_explicit_precise_ingredients:
                pi_id_or_ids: int | list[int] | None = next(listed)

                # For simplicity, convert to iterable
                pi_ids: list[int]
                if pi_id_or_ids is None:
                    pi_ids = []
                elif isinstance(pi_id_or_ids, int):
                    if not definition.ingredient_cardinality == Cardinality.ONE:
                        # Programming error
                        raise ValueError(
                            f"Expected single pi ID for "
                            f"{definition.class_id}, got {pi_id_or_ids}"
                        )
                    pi_ids = [pi_id_or_ids]
                else:
                    pi_ids = pi_id_or_ids

                # Lookup the atoms
                # NOTE: order for PI and I is expected to be the same; this is
                # future-proofing in any case, as only single-ingredient CDCs
                # can now have PIs
                nested_break: bool = False
                for pi_id in pi_ids:
                    # Find matching ingredient
                    for ing in explicit_ingredients:
                        try:
                            possible_pis = self.atoms.precise_ingredient[ing]
                            break
                        except KeyError:
                            # Not an error, try others
                            continue
                    else:  # No break
                        self.logger.debug(
                            f"Precise ingredient {pi_id} does not match any "
                            f"ingredient for {definition.class_id} {concept_id}"
                        )
                        nested_break = True
                        break

                    # Find the precise ingredient
                    try:
                        pi = possible_pis[dc.ConceptId(pi_id)]
                    except KeyError:
                        self.logger.debug(
                            f"Precise ingredient {pi_id} not found for "
                            f"{definition.class_id} {concept_id}"
                        )
                        node_bad_pi.append(concept_id)
                        nested_break = True
                        break

                    explicit_pis.append(pi)
                if nested_break:
                    continue

            ing_id_strength: Mapping[int, dc.Strength | None]
            try:
                if node_strength is not None:
                    ing_id_strength = {
                        ing: stg for ing, stg in node_strength[concept_id]
                    }
                elif node_ingreds is not None:
                    ing_id_strength = {
                        ing_id: None for ing_id in node_ingreds[concept_id]
                    }
                else:
                    # Programming error
                    raise ValueError(
                        f"Expected either strength or ingredient data for "
                        f"{definition.class_id} {concept_id}"
                    )
            except KeyError:
                # Filtered out by strength data parsing, so no need to report
                continue

            # Get bound strengths
            bound_strengths: list[
                dc.BoundStrength[dc.ConceptId, dc.Strength | None]
            ] = []
            if explicit_ingredients:
                # If node defines explicit ingredients, check them against DS
                cr_ings = sorted(
                    explicit_ingredients, key=lambda x: x.identifier
                )
                ds_ids = sorted(ing_id_strength.keys())

                if len(cr_ings) != len(ds_ids):
                    self.logger.debug(
                        f"Ingredient count mismatch for {definition.class_id} "
                        f"{concept_id}. Defined by CONCEPT_RELATIONSHIP: "
                        f"{cr_ings}; Defined by DRUG_STRENGTH: {ds_ids}."
                    )
                    node_ingred_ds_mismatch.append(concept_id)
                    continue

                nested_break = False
                for cr_ing, ds_id in zip(cr_ings, ds_ids):
                    if cr_ing.identifier != ds_id:
                        self.logger.debug(
                            f"Ingredient mismatch for {definition.class_id} "
                            f"{concept_id}. Defined by CONCEPT_RELATIONSHIP: "
                            f"{cr_ings}; Defined by DRUG_STRENGTH: {ds_ids}."
                        )
                        node_ingred_ds_mismatch.append(concept_id)
                        nested_break = True
                        break
                    bound_strengths.append((
                        cr_ing,
                        ing_id_strength[ds_id],
                    ))
                if nested_break:
                    continue
            else:
                # Obtain Ingredient atoms from DS data de novo
                nested_break = False
                for ing_id, stg in ing_id_strength.items():
                    try:
                        ing = self.atoms.ingredient[dc.ConceptId(ing_id)]
                    except KeyError:
                        self.logger.debug(
                            f"Ingredient {ing_id} not found for "
                            f"{definition.class_id} {concept_id}"
                        )
                        node_bad_ingred.append(concept_id)
                        nested_break = True
                        break
                    bound_strengths.append((ing, stg))
                if nested_break:
                    continue
            predicate_strength_data = SortedTuple(bound_strengths)

            # Iterate over parent definitions, test attributes gathered so far
            nested_break = False
            for p_def, nodes in parent_data.items():
                # Test attribute matches where required
                if p_def in p_def_a_def:
                    for attr_rel in d.MONO_ATTRIBUTE_DEFINITIONS.values():
                        a = attr_rel.target_definition
                        if a in p_def_a_def[p_def]:
                            if a not in attr_data:
                                # Programming error
                                raise ValueError(
                                    f"Parent {p_def.get_abbreviation()} "
                                    f"requires match on {a.class_id}, "
                                    f"but it is not defined for "
                                    f"{definition.class_id}"
                                )

                            for parent_node in nodes:
                                assert isinstance(a, d.MonoAtributeDefiniton)
                                parent_atom: _MonoAttribute = getattr(
                                    parent_node, a.node_getter
                                )()
                                own_atom = attr_data[a]
                                if parent_atom != own_atom:
                                    self.logger.debug(
                                        f"Attribute mismatch for "
                                        f"{definition.class_id} {concept_id}: "
                                        f"{own_atom} != {parent_atom} in "
                                        f"{p_def.class_id} "
                                        f"{parent_node.identifier}"
                                    )
                                    node_attr_mismatch.setdefault(
                                        p_def, {}
                                    ).setdefault(a, []).append(concept_id)
                                    nested_break = True
                                    break
                            if nested_break:
                                break
                        if nested_break:
                            break
                    if nested_break:
                        break
                if nested_break:
                    break

                # NOTE: Currently, there is no inheritance rules defined for
                # Precise Ingredients for any two classes, so there are no
                # checks or them

                # Test strength/ingredient matches
                if (
                    p_def.ingredient_cardinality
                    == definition.ingredient_cardinality
                ):
                    # Both are expected to be either mono or multicomponent
                    for parent_node in nodes:
                        parent_strength = parent_node.get_strength_data()
                        if len(parent_strength) != len(predicate_strength_data):
                            self.logger.debug(
                                f"Strength mismatch for {definition.class_id} "
                                f"{concept_id}: {len(parent_strength)} != "
                                f"{len(predicate_strength_data)} "
                                f"in {p_def.class_id} "
                                f"{parent_node.identifier}"
                            )
                            node_strength_mismatch.setdefault(p_def, []).append(
                                concept_id
                            )
                            nested_break = True
                            break

                        shared_iter = zip_longest(
                            parent_strength, predicate_strength_data
                        )
                        for (p_ing, p_stg), (o_ing, o_stg) in shared_iter:
                            if p_ing != o_ing:
                                self.logger.debug(
                                    f"Ingredient mismatch for "
                                    f"{definition.class_id} {concept_id}: "
                                    f"{p_ing} != {o_ing} in "
                                    f"{p_def.class_id} {parent_node.identifier}"
                                )
                                node_strength_mismatch.setdefault(
                                    p_def, []
                                ).append(concept_id)
                                nested_break = True
                                break

                            # Strength can be None
                            if p_stg is not None:
                                # Encountering a None strength in the child but
                                # not in the parent is a programming error
                                if o_stg is None:
                                    raise ValueError(
                                        f"Expected strength for "
                                        f"{definition.class_id} {concept_id} "
                                        f"to be non-None in {p_def.class_id} "
                                        f"{parent_node.identifier}"
                                    )

                                if not p_stg.matches(o_stg):
                                    self.logger.debug(
                                        f"Strength mismatch for "
                                        f"{definition.class_id} {concept_id}: "
                                        f"{p_stg} != {o_stg} in "
                                        f"{p_def.class_id} "
                                        f"{parent_node.identifier}"
                                    )
                                    node_strength_mismatch.setdefault(
                                        p_def, []
                                    ).append(concept_id)
                                    nested_break = True
                                    break
                        if nested_break:
                            break
                    if nested_break:
                        break
                else:  # Parent must be Cardinality.ONE, predicate -- NONZERO
                    if (
                        p_def.ingredient_cardinality != Cardinality.ONE
                        or definition.ingredient_cardinality
                        != Cardinality.NONZERO
                    ):
                        # Programming error
                        raise ValueError(
                            f"Expected parent {p_def.get_abbreviation()} and "
                            f"{definition.get_abbreviation()} to have "
                            f"opposite ingredient cardinalities"
                        )
                    for parent_node in nodes:
                        p_ing, p_stg = parent_node.get_strength_data()[0]
                        matched_any = False
                        for o_ing, o_stg in predicate_strength_data:
                            assert o_stg or (o_stg is None and p_stg is None)
                            ing_match = p_ing == o_ing
                            stg_match: bool
                            if p_stg is None:
                                stg_match = True
                            elif o_stg is not None:
                                stg_match = p_stg.matches(o_stg)
                            else:
                                # Programming error
                                raise ValueError(
                                    f"Expected strength for "
                                    f"{definition.class_id} {concept_id} "
                                    f"to be non-None for testing against "
                                    f"{p_def.class_id} {parent_node.identifier}"
                                )

                            if ing_match and stg_match:
                                matched_any = True
                                break
                        if not matched_any:
                            self.logger.debug(
                                f"Strength mismatch between "
                                f"{definition.class_id} {concept_id} and "
                                f"it's parent {p_def.class_id} "
                                f"{parent_node.identifier}: match for "
                                f"{p_ing}, {p_stg} not found in predicate"
                            )
                            node_strength_mismatch.setdefault(p_def, []).append(
                                concept_id
                            )
                            nested_break = True
                            break

                if nested_break:
                    break

                if p_def.defines_box_size:
                    if not definition.defines_box_size:
                        raise ValueError(
                            f"Parent {p_def.class_id} defines box size, but "
                            f"{definition.class_id} does not"
                        )
                    if node_box_size is None or concept_id not in node_box_size:
                        raise ValueError(
                            f"Expected box size for {definition.class_id} "
                            f"{concept_id}"
                        )
                    for parent_node in nodes:
                        parent_box_size = parent_node.get_box_size()
                        if parent_box_size != node_box_size[concept_id]:
                            self.logger.debug(
                                f"Box size mismatch for {definition.class_id} "
                                f"{concept_id}: {node_box_size} != "
                                f"{parent_box_size} in {p_def.class_id} "
                                f"{parent_node.identifier}"
                            )
                            node_strength_mismatch.setdefault(p_def, []).append(
                                concept_id
                            )
                            nested_break = True
                            break
                    if nested_break:
                        break

            if nested_break:
                continue  # Skip to next concept

            # Instantiate the class node
            node: dc.DrugNode[dc.ConceptId, dc.Strength | None]
            try:
                node = definition.constructor.from_definitions(  # pyright: ignore[reportUnknownMemberType] # noqa: E501
                    identifier=dc.ConceptId(concept_id),
                    parents={
                        p_def.omop_concept_class_id: nodes
                        for p_def, nodes in parent_data.items()
                    },
                    attributes={
                        a_def.omop_concept_class_id: a_atom
                        for a_def, a_atom in attr_data.items()
                    },
                    precise_ingredients=explicit_pis,
                    strength_data=predicate_strength_data,
                    box_size=node_box_size[concept_id]
                    if node_box_size
                    else None,
                )
            except RxConceptCreationError as e:
                self.logger.debug(
                    f"Failed to create {definition.class_id} {concept_id}: {e}"
                )
                node_failed.append(concept_id)
                continue

            node_idx = self.hierarchy.add_drug_node(node, parent_indices)
            out_nodes[concept_id] = node_idx

        # Cleanup
        # Bad attributes and parents
        for attr_def, lst_bad in chain(
            node_bad_attr.items(), node_bad_parent.items()
        ):
            self.filter_out_bad_concepts(
                len(node_concepts),
                pl.Series(lst_bad, dtype=pl.UInt32),
                f"All {definition.class_id} have valid {attr_def.class_id}",
                f"{definition.get_abbreviation()}_Bad_"
                f"{attr_def.get_abbreviation()}",
                f"{len(lst_bad):,} {definition.class_id} had bad "
                f"{attr_def.class_id}",
            )

        # Bad ingredient data
        self.filter_out_bad_concepts(
            len(node_concepts),
            pl.Series(node_bad_ingred, dtype=pl.UInt32),
            f"All {definition.class_id} have valid I",
            f"{definition.get_abbreviation()}_Bad_I",
            f"{len(node_bad_ingred):,} {definition.class_id} had bad "
            f"Ingredients",
        )

        # Strength mismatch
        if (
            definition.allowed_strength_configurations
            and definition.defines_explicit_ingredients
        ):
            self.filter_out_bad_concepts(
                len(node_concepts),
                pl.Series(node_ingred_ds_mismatch, dtype=pl.UInt32),
                f"All {definition.class_id} match Strengths to Ingredients",
                f"{definition.get_abbreviation()}_I_DS_Mismatch",
                f"{len(node_ingred_ds_mismatch):,} {definition.class_id} had "
                f"mismatched explicit Ingredients and Drug Strength data",
            )

        # Bad Precise Ingredient data
        if definition.defines_explicit_precise_ingredients:
            self.filter_out_bad_concepts(
                len(node_concepts),
                pl.Series(node_bad_pi, dtype=pl.UInt32),
                f"All {definition.class_id} have valid PI",
                f"{definition.get_abbreviation()}_Bad_PI",
                f"{len(node_bad_pi):,} {definition.class_id} had bad "
                f"Precise Ingredients",
            )

        # Mismatch with parents on attributes
        for p_def, attr_list in p_def_a_def.items():
            for attr_def in attr_list:
                mismatched = node_attr_mismatch.get(p_def, {}).get(attr_def, [])
                self.filter_out_bad_concepts(
                    len(node_concepts),
                    pl.Series(mismatched, dtype=pl.UInt32),
                    f"All {definition.class_id} have matching "
                    f"{attr_def.class_id} with their {p_def.class_id}s",
                    definition.get_abbreviation()
                    + "_"
                    + attr_def.get_abbreviation()
                    + "_Mismatch",
                    f"{len(mismatched):,} {definition.class_id} had mismatched "
                    f"{attr_def.class_id} with their {p_def.class_id}s",
                )

        # Mismatch with parents on ingredients and strengths
        for p_def in p_def_nodes:
            mismatched = node_strength_mismatch.get(p_def, [])
            self.filter_out_bad_concepts(
                len(node_concepts),
                pl.Series(mismatched, dtype=pl.UInt32),
                f"All {definition.class_id} have matching I and S with their "
                f"{p_def.class_id}s",
                definition.get_abbreviation()
                + "_"
                + p_def.get_abbreviation()
                + "_Mismatch",
                f"{len(mismatched):,} {definition.class_id} had mismatched "
                f"I and S with their {p_def.class_id}s",
            )

        return out_nodes
