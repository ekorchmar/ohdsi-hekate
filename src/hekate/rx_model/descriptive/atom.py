"""
Contains the definitions of the atomic attributes.
"""

from dataclasses import dataclass  # For shared characteristics
import rx_model.drug_classes as dc  # For constructor classes
from rx_model.descriptive.relationship import (
    RelationshipDescription,  # For mono-attribute relations
    Cardinality,  # For Cardinality.ONE
)
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
@dataclass(frozen=True, eq=True)
class MonoAtributeDefiniton(ConceptDefinition):
    """Shared behavior for mono-attribute definitions."""

    omop_domain_id: DomainId = DomainId.DRUG
    omop_vocabulary_ids: tuple[VocabularyId, ...] = RX_VOCAB
    standard_concept: bool = False


DOSE_FORM_DEFINITION = MonoAtributeDefiniton(
    omop_concept_class_id=ConceptClassId.DOSE_FORM,
    constructor=dc.DoseForm,
)

BRAND_NAME_DEFINITION = MonoAtributeDefiniton(
    omop_concept_class_id=ConceptClassId.BRAND_NAME,
    constructor=dc.BrandName,
)


SUPPLIER_DEFINITION = MonoAtributeDefiniton(
    omop_concept_class_id=ConceptClassId.SUPPLIER,
    omop_vocabulary_ids=(VocabularyId.RXE,),  # Only in RxNorm Extension
    constructor=dc.Supplier,
)

MONO_ATTRIBUTE_DEFINITIONS = {
    ConceptClassId.DOSE_FORM: RelationshipDescription(
        relationship_id="RxNorm has dose form",
        cardinality=Cardinality.ONE,
        target_class="DoseForm",
        target_definition=DOSE_FORM_DEFINITION,
    ),
    ConceptClassId.BRAND_NAME: RelationshipDescription(
        relationship_id="Has brand name",
        cardinality=Cardinality.ONE,
        target_class="BrandName",
        target_definition=PRECISE_INGREDIENT_DEFINITION,
    ),
    ConceptClassId.SUPPLIER: RelationshipDescription(
        relationship_id="Has supplier",
        cardinality=Cardinality.ONE,
        target_class="Supplier",
        target_definition=SUPPLIER_DEFINITION,
    ),
}

# Units
UNIT_DEFINITION = ConceptDefinition(
    omop_concept_class_id=ConceptClassId.UNIT,
    omop_domain_id=DomainId.UNIT,
    omop_vocabulary_ids=(VocabularyId.UCUM,),
    standard_concept=True,
    constructor=dc.Unit,
)
