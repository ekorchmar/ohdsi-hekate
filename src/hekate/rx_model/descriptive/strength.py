"""
Contains declarative data of strength configurations in OMOP CDM.
"""

from enum import Enum  # for string enums
from typing import Final  # for type hinting

import polars as pl  # for expressions
from utils.constants import PERCENT_CONCEPT_ID  # for strength configuration


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
