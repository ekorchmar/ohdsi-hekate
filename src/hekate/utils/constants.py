"""Constants for the project."""

import polars as pl

from typing import Literal

VALID_CONCEPT_END_DATE = 2099_12_31

# List of all concept_relationship_ids that are relevant to the project.
# See `reference/distinct_concept_relationship.*` for the source.
# PERF: We need to deduplicate this list and use the semantic child to semantic
# parent relationship
ALL_CONCEPT_RELATIONSHIP_IDS: pl.Series = pl.Series(
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

type DefiningMonoAttributeClass = Literal["Dose Form", "Brand Name", "Supplier"]

DEFINING_ATTRIBUTE_RELATIONSHIP: dict[DefiningMonoAttributeClass, str] = {
    "Dose Form": "RxNorm has dose form",
    "Brand Name": "Has brand name",
    "Supplier": "Has supplier",
}

REPLACEMENT_RELATIONSHIP = [
    "Maps to",
    "Mapped from",
    "Concept replaced by",
    "Concept replaces",
]

PERCENT_CONCEPT_ID = 8554
