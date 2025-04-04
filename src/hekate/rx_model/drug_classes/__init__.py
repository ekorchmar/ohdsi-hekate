"""
Contains the individual RxNorm and RxNorm Extension drug classes.

The classes do not implement any external integrity checks, e.g. verifying
consistency between DRUG_STRENGTH data for a Branded Drug and its Clinical Drug
ancestor.

However, if any checks can operate on class attributes themselves, they are
implemented. For example, the `SolidStrength` class checks that the values are
non-negative and not NaN.
"""

from rx_model.drug_classes.atom import (
    BrandName,
    DoseForm,
    Ingredient,
    Supplier,
    PreciseIngredient,
    Unit,
)

from rx_model.drug_classes.generic import (
    BoundStrength,
    ConceptIdentifier,
    ConceptCodeVocab,
    ConceptId,
    DrugNode,
)

from rx_model.drug_classes.strength import (
    SolidStrength,
    LiquidConcentration,
    LiquidQuantity,
    GasPercentage,
    Strength,
    UnquantifiedStrength,
)

from rx_model.drug_classes.complex import (
    ClinicalDrugComponent,
    ClinicalDrugForm,
    BrandedDrugComponent,
    BrandedDrugForm,
    ClinicalDrug,
    BrandedDrug,
    QuantifiedClinicalDrug,
    QuantifiedBrandedDrug,
    ClinicalDrugBox,
    QuantifiedClinicalBox,
    BrandedDrugBox,
    QuantifiedBrandedBox,
)
from rx_model.drug_classes.foreign import (
    BoundForeignStrength,
    PrecedenceData,
    ForeignDrugNode,
    ForeignNodePrototype,
    PseudoUnit,
    ForeignStrength,
)
from rx_model.drug_classes.relations import DRUG_CLASS_PREFERENCE_ORDER

__all__ = [
    "BoundStrength",
    "BoundForeignStrength",
    "BrandName",
    "BrandedDrug",
    "BrandedDrugBox",
    "BrandedDrugComponent",
    "BrandedDrugForm",
    "ClinicalDrug",
    "ClinicalDrugComponent",
    "ClinicalDrugBox",
    "ClinicalDrugForm",
    "ConceptCodeVocab",
    "ConceptId",
    "ConceptIdentifier",
    "DoseForm",
    "DrugNode",
    "DRUG_CLASS_PREFERENCE_ORDER",
    "ForeignDrugNode",
    "ForeignNodePrototype",
    "ForeignStrength",
    "PseudoUnit",
    "GasPercentage",
    "Ingredient",
    "LiquidConcentration",
    "LiquidQuantity",
    "PrecedenceData",
    "PreciseIngredient",
    "QuantifiedBrandedBox",
    "QuantifiedBrandedDrug",
    "QuantifiedClinicalBox",
    "QuantifiedClinicalDrug",
    "SolidStrength",
    "Strength",
    "Supplier",
    "Unit",
    "UnquantifiedStrength",
]
