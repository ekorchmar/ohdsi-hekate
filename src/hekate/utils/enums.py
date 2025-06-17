"""
Shared primitive (str, int) enums describing all kinds of valid states.
"""

from enum import Enum


class DomainId(Enum):
    DRUG = "Drug"
    UNIT = "Unit"


class VocabularyId(Enum):
    RXN = "RxNorm"
    RXE = "RxNorm Extension"
    UCUM = "UCUM"


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
    BP = "Branded Pack"

    # RxNorm Extension concept classes
    CDB = "Clinical Drug Box"
    BDB = "Branded Drug Box"
    QCB = "Quant Clinical Box"
    QBB = "Quant Branded Box"
    CPB = "Clinical Pack Box"
    BPB = "Branded Pack Box"
    # Pseudoclass
    MP = "Marketed Product"


class Cardinality(Enum):
    """
    Enum to define the cardinality of a relationship between two concepts

    Left hand side (source concept, concept_id_1) is always assumed to have
    cardinality of 1. Cardinality counts are always in relation to the target,
    showing how many target concepts can be related to a single source concept.
    """

    ANY = "0..*"  # Only used for pack contents
    ONE = "1..1"
    OPTIONAL = "0..1"
    NONZERO = "1..*"
    # Relationship that is ideally exists as 1..1, but can be 1..* in practice
    REDUNDANT = "1..?"
