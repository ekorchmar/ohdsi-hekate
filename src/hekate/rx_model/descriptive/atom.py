"""
Contains the definitions of the atomic attributes.
"""

from dataclasses import dataclass  # For shared characteristics
import rx_model.drug_classes as dc  # For constructor classes
from rx_model.descriptive.base import (
    RX_VOCAB,
    ConceptClassId,
    ConceptDefinition,
    DomainId,
    VocabularyId,
)


# RxN/E Ingredients
INGREDIENT_DEFINITION = ConceptDefinition(
    omop_concept_class_id=ConceptClassId.INGREDIENT,
    omop_domain_id=DomainId.DRUG,
    omop_vocabulary_ids=RX_VOCAB,
    standard_concept=True,
    constructor=dc.Ingredient,
)

PRECISE_INGREDIENT_DEFINITION = ConceptDefinition(
    omop_concept_class_id=ConceptClassId.PRECISE_INGREDIENT,
    omop_domain_id=DomainId.DRUG,
    omop_vocabulary_ids=(VocabularyId.RXN,),  # For now, only in RxNorm
    standard_concept=False,
    constructor=dc.PreciseIngredient,
)


# Rxn/E mono attributes
@dataclass
class MonoAtributeDefiniton(ConceptDefinition):
    """Shared behavior for mono-attribute definitions."""

    omop_domain_id: DomainId = DomainId.DRUG
    omop_vocabulary_ids: tuple[VocabularyId, ...] = RX_VOCAB
    defining_relationship_id: str | None = None
    standard_concept: bool = False

    def __post_init__(self):
        # Self check
        if self.defining_relationship_id is None:
            raise ValueError("Missing defining relationship ID")


DOSE_FORM_DEFINITION = MonoAtributeDefiniton(
    omop_concept_class_id=ConceptClassId.DOSE_FORM,
    constructor=dc.DoseForm,
    defining_relationship_id="RxNorm has dose form",
)

BRAND_NAME_DEFINITION = MonoAtributeDefiniton(
    omop_concept_class_id=ConceptClassId.BRAND_NAME,
    constructor=dc.BrandName,
    defining_relationship_id="Has brand name",
)


SUPPLIER_DEFINITION = MonoAtributeDefiniton(
    omop_concept_class_id=ConceptClassId.SUPPLIER,
    omop_vocabulary_ids=(VocabularyId.RXE,),  # Only in RxNorm Extension
    constructor=dc.Supplier,
    defining_relationship_id="Has supplier",
)

MONO_ATTRIBUTE_DEFINITIONS = [
    DOSE_FORM_DEFINITION,
    BRAND_NAME_DEFINITION,
    SUPPLIER_DEFINITION,
]

# Units
UNIT_DEFINITION = ConceptDefinition(
    omop_concept_class_id=ConceptClassId.UNIT,
    omop_domain_id=DomainId.UNIT,
    omop_vocabulary_ids=(VocabularyId.UCUM,),
    standard_concept=True,
    constructor=dc.Unit,
)
