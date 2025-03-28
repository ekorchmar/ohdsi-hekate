"""
Contains base classes for declarative OMOP representation of RxNorm and
RxNorm Extension concepts.
"""

from dataclasses import dataclass  # For concept definitions
from enum import Enum  # For string enums

import polars as pl  # For expression definitions


class DomainId(Enum):
    DRUG = "Drug"
    UNIT = "Unit"


class VocabularyId(Enum):
    RXN = "RxNorm"
    RXE = "RxNorm Extension"
    UCUM = "UCUM"


RX_VOCAB = VocabularyId.RXE, VocabularyId.RXN


class ConceptClassId(Enum):
    # RxNorm atoms
    INGREDIENT = "Ingredient"
    DOSE_FORM = "Dose Form"
    BRAND_NAME = "Brand Name"
    PRECISE_INGREDIENT = "Precise Ingredient"
    # RxNorm Extension atoms
    SUPPLIER = "Supplier"
    # UCUM atoms
    UNIT = "Unit"

    # RxNorm-native concept classes
    CDC = "Clinical Drug Comp"
    BDC = "Branded Drug Comp"
    CDF = "Clinical Drug Form"
    BDF = "Branded Drug Form"
    CD = "Clinical Drug"
    BD = "Branded Drug"
    QCD = "Quant Clinical Drug"
    QBD = "Quant Branded Drug"
    CP = "Clinical Pack"

    # RxNorm Extension concept classes
    BP = "Branded Pack"
    CDB = "Clinical Drug Box"
    BDB = "Branded Drug Box"
    QCB = "Quant Clinical Box"
    QBB = "Quant Branded Box"
    CPB = "Clinical Pack Box"
    BPB = "Branded Pack Box"
    # Pseudoclass
    MP = "Marketed Product"


@dataclass(frozen=True, eq=True)
class ConceptDefinition:
    constructor: type
    omop_concept_class_id: ConceptClassId
    omop_domain_id: DomainId
    omop_vocabulary_ids: tuple[VocabularyId, ...]
    standard_concept: bool

    def get_concept_expression(self) -> pl.Expr:
        return (
            (pl.col("concept_class_id") == self.omop_concept_class_id.value)
            & (pl.col("domain_id") == self.omop_domain_id.value)
            & (
                (pl.col("standard_concept") == "S")
                if self.standard_concept
                else pl.col("standard_concept").is_null()
            )
            & (
                pl.col("vocabulary_id").is_in([
                    vocabulary_id.value
                    for vocabulary_id in self.omop_vocabulary_ids
                ])
            )
        )

    def get_abbreviation(self) -> str:
        return "".join(
            word[0].upper() for word in self.omop_concept_class_id.value.split()
        )

    def get_colname(self) -> str:
        return (
            "_".join(self.omop_concept_class_id.value.lower().split())
            + "_concept_id"
        )

    @property
    def class_id(self) -> str:
        return self.omop_concept_class_id.value
