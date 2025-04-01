"""
Contains definitions for the RxNorm and RxNorm Extension Pack classes.
"""

from __future__ import annotations

from dataclasses import dataclass  # For dataclass definitions
from typing import ClassVar  # For registry

import rx_model.drug_classes as dc  # For drug classes
from rx_model.descriptive.base import (
    RX_VOCAB,
    ConceptDefinition,  # For content definitions
)
from rx_model.descriptive.complex import (
    ComplexDrugNodeDefinition,  # For content definitions
)
from rx_model.descriptive.relationship import (
    RelationshipDescription,  # For parent and content relations
)
from utils.enums import ConceptClassId, DomainId, VocabularyId  # For enums


@dataclass(frozen=True, eq=True)
class PackDefinition(ConceptDefinition):
    """
    Class representing the RxNorm and RxNorm Extension Pack classes.
    """

    constructor: type[dc.PackNode[dc.ConceptId]]
    content_definitions: tuple[ComplexDrugNodeDefinition, ...]
    attribute_definitions: tuple[RelationshipDescription, ...]

    defines_pack_size: bool

    omop_domain_id: DomainId = DomainId.DRUG
    omop_vocabulary_ids: tuple[VocabularyId, ...] = RX_VOCAB  # RxE only for Box
    standard_concept: bool = True

    # Registry
    _registry: ClassVar[dict[ConceptClassId, PackDefinition]] = {}

    @classmethod
    def get(cls, key: ConceptClassId) -> PackDefinition:
        return cls._registry[key]

    def __post_init__(self):
        # TODO: integrity checks
        self._registry[self.omop_concept_class_id] = self
