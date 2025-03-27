"""Constants for the project."""

from enum import Enum
import polars as pl

from typing import Final, Literal

VALID_CONCEPT_START_DATE: Literal[1970_01_01] = 1970_01_01
VALID_CONCEPT_END_DATE: Literal[2099_12_31] = 2099_12_31

# List of all concept_relationship_ids that are relevant to the project.
# See `reference/distinct_concept_relationship.*` for the source.
# PERF: We need to deduplicate this list and use the semantic child to semantic
# parent relationship
ALL_CONCEPT_RELATIONSHIP_IDS: Final[pl.Series] = pl.Series(
    [
        "Available as box",
        "Box of",
        "Brand name of",
        "Component of",
        "Consists of",
        "Constitutes",
        "Contained in",
        "Contains",
        "Dose form group of",
        "Drug class of drug",
        "Drug has drug class",
        "Form of",
        "Has brand name",
        "Has component",
        "Has dose form group",
        "Has form",
        "Has marketed form",
        "Has precise ing",
        "Has quantified form",
        "Has supplier",
        "Has tradename",
        "Mapped from",
        "Maps to",
        "Marketed form of",
        "Precise ing of",
        "Quantified form of",
        "RxNorm dose form of",
        "RxNorm has dose form",
        "RxNorm has ing",
        "RxNorm ing of",
        "RxNorm inverse is a",
        "RxNorm is a",
        "Supplier of",
        "Tradename of",
    ],
    dtype=pl.Utf8,
)

type DefiningMonoAttribute = Literal["Dose Form", "Brand Name", "Supplier"]

DEFINING_ATTRIBUTE_RELATIONSHIP: Final[dict[DefiningMonoAttribute, str]] = {
    "Dose Form": "RxNorm has dose form",
    "Brand Name": "Has brand name",
    "Supplier": "Has supplier",
}

REPLACEMENT_RELATIONSHIP: Final[list[str]] = [
    "Maps to",
    "Mapped from",
    "Concept replaced by",
    "Concept replaces",
]

PERCENT_CONCEPT_ID: Literal[8554] = 8554


class StrengthConfiguration(Enum):
    AMOUNT_ONLY = "AMOUNT_ONLY"
    LIQUID_CONCENTRATION = "LIQUID_CONCENTRATION"
    LIQUID_QUANTITY = "LIQUID_QUANTITY"
    GAS_PERCENTAGE = "GAS_PERCENTAGE"


STRENGTH_CONFIGURATIONS_ID: Final[dict[StrengthConfiguration, pl.Expr]] = {
    # - Amount value and unit are present, rest of the fields are null
    StrengthConfiguration.AMOUNT_ONLY: (
        pl.col("amount_value").is_not_null()
        & pl.col("amount_unit_concept_id").is_not_null()
        & pl.col("numerator_value").is_null()
        & pl.col("numerator_unit_concept_id").is_null()
        & pl.col("denominator_value").is_null()
        & pl.col("denominator_unit_concept_id").is_null()
    ),
    # - Numerator value and unit are present, but denominator is unit only
    StrengthConfiguration.LIQUID_CONCENTRATION: (
        pl.col("amount_value").is_null()
        & pl.col("amount_unit_concept_id").is_null()
        & pl.col("numerator_value").is_not_null()
        & pl.col("numerator_unit_concept_id").is_not_null()
        & pl.col("denominator_value").is_null()
        & pl.col("denominator_unit_concept_id").is_not_null()
    ),
    # - Numerator and denominator values and units are present
    StrengthConfiguration.LIQUID_QUANTITY: (
        pl.col("amount_value").is_null()
        & pl.col("amount_unit_concept_id").is_null()
        & pl.col("numerator_value").is_not_null()
        & pl.col("numerator_unit_concept_id").is_not_null()
        & pl.col("denominator_value").is_not_null()
        & pl.col("denominator_unit_concept_id").is_not_null()
    ),
    # - Gases are weird. They can have numerator with percents exactly in
    #   numerator and no denominator data.
    StrengthConfiguration.GAS_PERCENTAGE: (
        pl.col("amount_value").is_null()
        & pl.col("amount_unit_concept_id").is_null()
        & pl.col("numerator_value").is_not_null()
        & (pl.col("numerator_unit_concept_id") == PERCENT_CONCEPT_ID)  # (%)
        & pl.col("denominator_value").is_null()
        & pl.col("denominator_unit_concept_id").is_null()
    ),
}

STRENGTH_CONFIGURATIONS_CODE: Final[dict[StrengthConfiguration, pl.Expr]] = {
    # - Amount value and unit are present, rest of the fields are null
    StrengthConfiguration.AMOUNT_ONLY: (
        pl.col("amount_value").is_not_null()
        & pl.col("amount_unit").is_not_null()
        & pl.col("numerator_value").is_null()
        & pl.col("numerator_unit").is_null()
        & pl.col("denominator_value").is_null()
        & pl.col("denominator_unit").is_null()
    ),
    # - Numerator value and unit are present, but denominator is unit only
    StrengthConfiguration.LIQUID_CONCENTRATION: (
        pl.col("amount_value").is_null()
        & pl.col("amount_unit").is_null()
        & pl.col("numerator_value").is_not_null()
        & pl.col("numerator_unit").is_not_null()
        & pl.col("denominator_value").is_null()
        & pl.col("denominator_unit").is_not_null()
    ),
    # - Numerator and denominator values and units are present
    StrengthConfiguration.LIQUID_QUANTITY: (
        pl.col("amount_value").is_null()
        & pl.col("amount_unit").is_null()
        & pl.col("numerator_value").is_not_null()
        & pl.col("numerator_unit").is_not_null()
        & pl.col("denominator_value").is_not_null()
        & pl.col("denominator_unit").is_not_null()
    ),
    # - Gases are weird. They can have numerator with percents exactly in
    #   numerator and no denominator data.
    StrengthConfiguration.GAS_PERCENTAGE: (
        pl.col("amount_value").is_null()
        & pl.col("amount_unit").is_null()
        & pl.col("numerator_value").is_not_null()
        & (pl.col("numerator_unit") == "%")
        & pl.col("denominator_value").is_null()
        & pl.col("denominator_unit").is_null()
    ),
    # TODO: Box size variations, once we start using them
}

_STRENGTH_CORRIDOR: Final[float] = 0.05
STRENGTH_CLOSURE_BOUNDARY_LOW: Final[float] = 1 - _STRENGTH_CORRIDOR
STRENGTH_CLOSURE_BOUNDARY_HIGH: Final[float] = 1 / (1 - _STRENGTH_CORRIDOR)


# TODO: separate flags for run configuration to a separate file

# Flag to stop the athena filtering on suspected overfiltering
ATHENA_OVERFILTERING_TRESHOLD: Final[float] = 0.30
ATHENA_OVERFILTERING_WARNING: Final[bool] = True

# NOTE: taken from:
# https://github.com/OHDSI/Vocabulary-v5.0/blob/master/working/packages/vocabulary_pack/BuildRxE.sql#L4487
UNIT_NAMES: Final[dict[int, str]] = {
    45744815: "ORGANISMS",
    8554: "%",
    45744809: "ACTUAT",
    45744811: "AU",
    45744810: "BAU",
    45744812: "CELLS",
    9278: "CFU",
    45744813: "CU",
    8505: "HR",
    8718: "IU",
    45744814: "LFU",
    44819154: "MCI",
    9551: "MEQ",
    8576: "MG",
    9367: "MIN",
    8587: "ML",
    9573: "MMOL",
    9439: "MU",
    9379: "PFU",
    45744816: "PNU",
    9483: "SQCM",
    9414: "TCID",
    8510: "UNT",
    9693: "IR",
    9325: "X",
    9324: "C",
}

BOX_SIZE_LIMIT: Final[int] = 2**15 - 1  # Postgres smallint limit
