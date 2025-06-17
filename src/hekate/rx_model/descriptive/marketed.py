"""
Contains definitions for Marketed Product pseudo-class.
"""

from __future__ import annotations


from rx_model import drug_classes as dc  # for parent relations
from rx_model.descriptive.atom import (
    MONO_ATTRIBUTE_RELATIONS,  # For mono-attribute relations
    BRAND_NAME_DEFINITION,  # For brand name attribute (optional in MP)
)
from rx_model.descriptive.base import (
    ConceptDefinition,
)
from rx_model.descriptive.relationship import (
    RelationshipDescription,  # For parent relations
)
from utils.enums import Cardinality, ConceptClassId, DomainId, VocabularyId

ALLOWED_MARKETED_CONTENT: list[ConceptClassId] = [
    # Drug content
    ConceptClassId.CD,
    ConceptClassId.BD,
    ConceptClassId.QCD,
    ConceptClassId.QBD,
    ConceptClassId.CDB,
    ConceptClassId.BDB,
    ConceptClassId.QCB,
    ConceptClassId.QBB,
    # Pack content
    ConceptClassId.CP,
    ConceptClassId.BP,
    # NOTE: Boxes are technically allowed, but Build_RxE does not create them
    ConceptClassId.CPB,
    ConceptClassId.BPB,
]

MARKETED_PRODUCT_DEFINITION = ConceptDefinition(
    constructor=dc.MarketedProductNode,
    omop_concept_class_id=ConceptClassId.MP,
    omop_domain_id=DomainId.DRUG,
    omop_vocabulary_ids=(VocabularyId.RXE,),  # RxE only
    standard_concept=True,
)

MARKETED_PRODUCT_RELATIONS: list[RelationshipDescription] = [
    MONO_ATTRIBUTE_RELATIONS[ConceptClassId.SUPPLIER],
    RelationshipDescription(
        relationship_id="Has brand name",
        cardinality=Cardinality.OPTIONAL,
        target_definition=BRAND_NAME_DEFINITION,
    ),
    # TODO: research DOSE FORM
]
