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
from rx_model.descriptive.atom import (
    MONO_ATTRIBUTE_RELATIONS,  # For attribute definitions
    DOSE_FORM_DEFINITION,  # For negative checks
)
from rx_model.descriptive.complex import (
    ComplexDrugNodeDefinition,  # For content definitions
)
from rx_model.descriptive.relationship import (
    RelationshipDescription,  # For parent and content relations
)
from utils.enums import (  # For enums
    Cardinality,
    ConceptClassId,
    DomainId,
    VocabularyId,
)


@dataclass(frozen=True, eq=True)
class PackDefinition(ConceptDefinition):
    """
    Class representing the RxNorm and RxNorm Extension Pack classes.
    """

    constructor: type[dc.PackNode[dc.ConceptId]]
    content_relations: tuple[RelationshipDescription, ...] = ()
    attribute_relations: tuple[RelationshipDescription, ...] = ()
    parent_relations: tuple[RelationshipDescription, ...] = ()

    defines_pack_size: bool = False

    omop_domain_id: DomainId = DomainId.DRUG
    omop_vocabulary_ids: tuple[VocabularyId, ...] = RX_VOCAB  # RxE only for Box
    standard_concept: bool = True

    # Registry
    registry: ClassVar[dict[ConceptClassId, PackDefinition]] = {}

    @classmethod
    def get(cls, key: ConceptClassId) -> PackDefinition:
        return cls.registry[key]

    def __post_init__(self):
        # Pack size is only defined for *Boxes, and is as such specific to RxE
        if self.defines_pack_size and self.omop_vocabulary_ids != (
            VocabularyId.RXE,
        ):
            raise ValueError("Pack size is specific for RxE packs!")

        # NOTE: RxN packs seem to define dose form of 19127776 (Pack), but it is
        # meaningless and will not be considered for integrity checks.

        # Dose form is not a valid attribute for packs
        if any(
            rel_id.target_definition == DOSE_FORM_DEFINITION
            for rel_id in self.attribute_relations
        ):
            raise ValueError("Dose form is not a valid attribute for packs!")

        # Only pack nodes can be parents
        if any(
            not issubclass(rel_id.target_definition.constructor, dc.PackNode)
            for rel_id in self.parent_relations
        ):
            raise ValueError("Only pack nodes can be parents of packs!")

        self.registry[self.omop_concept_class_id] = self


# Declarations of pack classes
_CP_DEFINITION = PackDefinition(
    constructor=dc.ClinicalPack,
    omop_concept_class_id=ConceptClassId.CP,
    content_relations=tuple(
        RelationshipDescription(
            relationship_id="Contains",
            target_definition=ComplexDrugNodeDefinition.get(cd_class),
            cardinality=Cardinality.ANY,
        )
        for cd_class in [ConceptClassId.CD, ConceptClassId.QCD]
    ),
    attribute_relations=(),
    parent_relations=(),
    defines_pack_size=False,
)

_BP_DEFINITION = PackDefinition(
    constructor=dc.BrandedPack,
    omop_concept_class_id=ConceptClassId.BP,
    content_relations=tuple(
        RelationshipDescription(
            relationship_id="Contains",
            target_definition=ComplexDrugNodeDefinition.get(cd_class),
            cardinality=Cardinality.ANY,
        )
        for cd_class in [
            # RxNorm
            ConceptClassId.BD,
            ConceptClassId.QBD,
            # RxNorm Extension
            ConceptClassId.CD,
            ConceptClassId.QCD,
        ]
    ),
    attribute_relations=(MONO_ATTRIBUTE_RELATIONS[ConceptClassId.BRAND_NAME],),
    parent_relations=(
        RelationshipDescription(
            relationship_id="Tradename of",
            target_definition=_CP_DEFINITION,
            cardinality=Cardinality.ONE,
        ),
    ),
    defines_pack_size=False,
)
