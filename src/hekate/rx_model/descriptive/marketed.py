"""
Contains definitions for Marketed Product pseudo-class.
"""

from __future__ import annotations


from rx_model.drug_classes.marketed import (
    MarketedProductNode,
)  # for parent relations
from rx_model.descriptive.base import ConceptDefinition
from utils.enums import ConceptClassId, DomainId, VocabularyId

MARKETED_PRODUCT_DEFINITION = ConceptDefinition(
    constructor=MarketedProductNode,
    omop_concept_class_id=ConceptClassId.MP,
    omop_domain_id=DomainId.DRUG,
    omop_vocabulary_ids=(VocabularyId.RXE,),  # RxE only
    standard_concept=True,
)
