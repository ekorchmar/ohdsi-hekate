"""
Contains definitions for complex drug classes, including declaration of their
relationships to other classes.
"""

from __future__ import annotations

from dataclasses import dataclass  # For shared characteristics
from typing import ClassVar  # For registry
from rx_model import drug_classes as dc  # For class constructors
from rx_model.descriptive.base import (
    RX_VOCAB,
    ConceptDefinition,
)
from rx_model.descriptive.atom import MONO_ATTRIBUTE_DEFINITIONS
from rx_model.descriptive.relationship import (
    RelationshipDescription,  # For parent relations
    CARDINALITY_REQUIRED,
)
from rx_model.descriptive.strength import (
    StrengthConfiguration,  # For strength configurations
    UNQUANTIFIED_STRENGTH_CONFIGURATIONS,
)
from utils.enums import Cardinality, DomainId, VocabularyId, ConceptClassId


@dataclass(frozen=True, eq=True)
class ComplexDrugNodeDefinition(ConceptDefinition):
    """Shared behavior for complex drug node definitions."""

    constructor: type[dc.DrugNode]  # pyright: ignore[reportMissingTypeArgument]
    attribute_definitions: tuple[RelationshipDescription, ...] = ()
    parent_relations: tuple[RelationshipDescription, ...] = ()
    allowed_strength_configurations: tuple[StrengthConfiguration, ...] = ()
    ingredient_cardinality: Cardinality = Cardinality.ANY

    # Boolean flags
    defines_explicit_ingredients: bool = False
    defines_explicit_precise_ingredients: bool = False  # Usually
    defines_box_size: bool = False  # Usually

    omop_domain_id: DomainId = DomainId.DRUG
    omop_vocabulary_ids: tuple[VocabularyId, ...] = RX_VOCAB  # RxE for some
    standard_concept: bool = True

    # Registry
    _registry: ClassVar[dict[ConceptClassId, ComplexDrugNodeDefinition]] = {}

    @classmethod
    def get(cls, key: ConceptClassId) -> ComplexDrugNodeDefinition:
        return cls._registry[key]

    def __post_init__(self):
        if self.ingredient_cardinality not in CARDINALITY_REQUIRED:
            raise ValueError(
                "Ingredient cardinality must be either ONE or NONZERO"
            )

        if (
            self.defines_box_size
            and len(self.allowed_strength_configurations) == 0
        ):
            raise ValueError(
                "Box size requires at least one strength configuration"
            )

        for relation in self.parent_relations:
            if relation.cardinality not in CARDINALITY_REQUIRED:
                raise ValueError(
                    "Parent relations must have a cardinality of either ONE or "
                    "NONZERO"
                )

        if (
            self.defines_explicit_precise_ingredients
            and not self.defines_explicit_ingredients
        ):
            raise ValueError(
                "Precise ingredients can only be defined if explicit "
                "ingredients are"
            )

        if (
            len(self.parent_relations) == 0
            and not self.defines_explicit_ingredients
        ):
            raise ValueError(
                "Complex drug nodes must have at least one parent relation or "
                "explicit ingredients"
            )

        self._registry[self.omop_concept_class_id] = self


# Declarations of complex drug classes
# NOTE: Those are not re-exported and should be retrieved with get()
_CDF_DEFINITION = ComplexDrugNodeDefinition(
    constructor=dc.ClinicalDrugForm,
    omop_concept_class_id=ConceptClassId.CDF,
    attribute_definitions=(
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.DOSE_FORM],
    ),
    parent_relations=(),
    allowed_strength_configurations=(),
    ingredient_cardinality=Cardinality.NONZERO,
    defines_explicit_ingredients=True,
)

_CDC_DEFINITION = ComplexDrugNodeDefinition(
    constructor=dc.ClinicalDrugComponent,
    omop_concept_class_id=ConceptClassId.CDC,
    attribute_definitions=(),
    parent_relations=(),
    allowed_strength_configurations=UNQUANTIFIED_STRENGTH_CONFIGURATIONS,
    ingredient_cardinality=Cardinality.ONE,
    defines_explicit_ingredients=True,
    defines_explicit_precise_ingredients=True,  # Only for CDCs!
)

_BDC_DEFINITION = ComplexDrugNodeDefinition(
    constructor=dc.BrandedDrugComponent,
    omop_concept_class_id=ConceptClassId.BDC,
    attribute_definitions=(
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.BRAND_NAME],
    ),
    parent_relations=(
        RelationshipDescription(
            relationship_id="Tradename of",
            cardinality=Cardinality.NONZERO,
            target_definition=_CDC_DEFINITION,
        ),
    ),
    allowed_strength_configurations=UNQUANTIFIED_STRENGTH_CONFIGURATIONS,
    ingredient_cardinality=Cardinality.NONZERO,
    defines_explicit_ingredients=False,
)

_CD_DEFINITION = ComplexDrugNodeDefinition(
    constructor=dc.ClinicalDrug,
    omop_concept_class_id=ConceptClassId.CD,
    attribute_definitions=(
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.DOSE_FORM],
    ),
    parent_relations=(
        RelationshipDescription(
            target_definition=_CDF_DEFINITION,
            relationship_id="Consists of",
            cardinality=Cardinality.ONE,
        ),
        RelationshipDescription(
            target_definition=_CDC_DEFINITION,
            relationship_id="RxNorm has a",
            cardinality=Cardinality.NONZERO,
        ),
    ),
    allowed_strength_configurations=UNQUANTIFIED_STRENGTH_CONFIGURATIONS,
    ingredient_cardinality=Cardinality.NONZERO,
    defines_explicit_ingredients=False,
    defines_box_size=False,
)

_BDF_DEFINITION = ComplexDrugNodeDefinition(
    constructor=dc.BrandedDrugForm,
    omop_concept_class_id=ConceptClassId.BDF,
    attribute_definitions=(
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.DOSE_FORM],
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.BRAND_NAME],
    ),
    parent_relations=(
        RelationshipDescription(
            target_definition=_CDF_DEFINITION,
            relationship_id="Tradename of",
            cardinality=Cardinality.ONE,
        ),
    ),
    allowed_strength_configurations=(),
    ingredient_cardinality=Cardinality.NONZERO,
    defines_explicit_ingredients=False,
    defines_box_size=False,
)

_BD_DEFINITION = ComplexDrugNodeDefinition(
    constructor=dc.BrandedDrug,
    omop_concept_class_id=ConceptClassId.BD,
    attribute_definitions=(
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.BRAND_NAME],
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.DOSE_FORM],
    ),
    parent_relations=(
        RelationshipDescription(
            target_definition=_BDC_DEFINITION,
            relationship_id="Consists of",
            cardinality=Cardinality.ONE,
        ),
        RelationshipDescription(
            relationship_id="Tradename of",
            cardinality=Cardinality.ONE,
            target_definition=_CD_DEFINITION,
        ),
        RelationshipDescription(
            relationship_id="RxNorm is a",
            cardinality=Cardinality.ONE,
            target_definition=_BDF_DEFINITION,
        ),
    ),
    allowed_strength_configurations=UNQUANTIFIED_STRENGTH_CONFIGURATIONS,
    ingredient_cardinality=Cardinality.NONZERO,
    defines_explicit_ingredients=False,
    defines_box_size=False,
)

_QCD_DEFINITION = ComplexDrugNodeDefinition(
    constructor=dc.QuantifiedClinicalDrug,
    omop_concept_class_id=ConceptClassId.QCD,
    attribute_definitions=(
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.DOSE_FORM],
    ),
    parent_relations=(
        RelationshipDescription(
            target_definition=_CD_DEFINITION,
            relationship_id="Quantified form of",
            cardinality=Cardinality.ONE,
        ),
    ),
    allowed_strength_configurations=(StrengthConfiguration.LIQUID_QUANTITY,),
    ingredient_cardinality=Cardinality.NONZERO,
    defines_explicit_ingredients=False,
    defines_box_size=False,
)

_QBD_DEFINITION = ComplexDrugNodeDefinition(
    constructor=dc.QuantifiedBrandedDrug,
    omop_concept_class_id=ConceptClassId.QBD,
    attribute_definitions=(
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.BRAND_NAME],
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.DOSE_FORM],
    ),
    parent_relations=(
        RelationshipDescription(
            target_definition=_BD_DEFINITION,
            relationship_id="Quantified form of",
            cardinality=Cardinality.ONE,
        ),
        RelationshipDescription(
            target_definition=_QCD_DEFINITION,
            relationship_id="Tradename of",
            cardinality=Cardinality.ONE,
        ),
    ),
    allowed_strength_configurations=(StrengthConfiguration.LIQUID_QUANTITY,),
    ingredient_cardinality=Cardinality.NONZERO,
    defines_explicit_ingredients=False,
    defines_box_size=False,
)

_CDB_DEFINITION = ComplexDrugNodeDefinition(
    constructor=dc.ClinicalDrugBox,
    omop_concept_class_id=ConceptClassId.CDB,
    attribute_definitions=(
        MONO_ATTRIBUTE_DEFINITIONS[ConceptClassId.DOSE_FORM],
    ),
    parent_relations=(
        RelationshipDescription(
            target_definition=_CD_DEFINITION,
            relationship_id="Box of",
            cardinality=Cardinality.ONE,
        ),
    ),
    allowed_strength_configurations=UNQUANTIFIED_STRENGTH_CONFIGURATIONS,
    ingredient_cardinality=Cardinality.NONZERO,
    defines_explicit_ingredients=False,
    defines_box_size=True,
)
