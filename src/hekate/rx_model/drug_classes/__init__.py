"""
Contains the individual RxNorm and RxNorm Extension drug classes.

The classes do not implement any external integrity checks, e.g. verifying
consistency between DRUG_STRENGTH data for a Branded Drug and its Clinical Drug
ancestor.

However, if any checks can operate on class attributes themselves, they are
implemented. For example, the `SolidStrength` class checks that the values are
non-negative and not NaN.
"""

from rx_model.drug_classes.base import (
    ConceptIdentifier,
    ConceptId,
    ConceptCodeVocab,
    HierarchyNode,
)

from rx_model.drug_classes.atom import (
    BrandName,
    DoseForm,
    Ingredient,
    Supplier,
    PreciseIngredient,
    Unit,
)

from rx_model.drug_classes.generic import (
    PackEntry,
    DrugNode,
    PackNode,
)

from rx_model.drug_classes.strength import (
    SolidStrength,
    LiquidConcentration,
    LiquidQuantity,
    GasPercentage,
    Strength,
    UnquantifiedStrength,
    BoundStrength,
    Concentration,
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

from rx_model.drug_classes.pack import (
    ClinicalPack,
    BrandedPack,
    ClinicalPackBox,
    BrandedPackBox,
)

from rx_model.drug_classes.foreign import (
    BoundForeignStrength,
    PrecedenceData,
    ForeignDrugNode,
    ForeignNodePrototype,
    PseudoUnit,
    ForeignStrength,
    ForeignPackNode,
)
from rx_model.drug_classes.relations import DRUG_CLASS_PREFERENCE_ORDER

__all__ = [
    "BoundForeignStrength",
    "BoundStrength",
    "BrandedDrug",
    "BrandedDrugBox",
    "BrandedDrugComponent",
    "BrandedDrugForm",
    "BrandedPack",
    "BrandedPackBox",
    "BrandName",
    "ClinicalDrug",
    "ClinicalDrugBox",
    "ClinicalDrugComponent",
    "ClinicalDrugForm",
    "ClinicalPack",
    "ClinicalPackBox",
    "Concentration",
    "ConceptCodeVocab",
    "ConceptId",
    "ConceptIdentifier",
    "DoseForm",
    "DRUG_CLASS_PREFERENCE_ORDER",
    "DrugNode",
    "ForeignDrugNode",
    "ForeignNodePrototype",
    "ForeignPackNode",
    "ForeignStrength",
    "GasPercentage",
    "HierarchyNode",
    "Ingredient",
    "LiquidConcentration",
    "LiquidQuantity",
    "PackEntry",
    "PackNode",
    "PrecedenceData",
    "PreciseIngredient",
    "PseudoUnit",
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
