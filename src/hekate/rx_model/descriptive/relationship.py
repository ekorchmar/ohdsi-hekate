"""
Contains OMOP representation of intra- RxNorm and RxNorm Extension
relationships.
"""

from enum import Enum  # For string enums
from typing import NamedTuple
from rx_model.descriptive.base import ConceptDefinition


class Cardinality(Enum):
    """
    Enum to define the cardinality of a relationship between two concepts

    Left hand side (source concept, concept_id_1) is always assumed to have
    cardinality of 1. Cardinality counts are always in relation to the target,
    showing how many target concepts can be related to a single source concept.
    """

    ANY = "0..*"  # Will not be used in practice
    ONE = "1..1"
    OPTIONAL = "0..1"
    NONZERO = "1..*"
    # Relationship that is ideally exists as 1..1, but can be 1..* in practice
    REDUNDANT = "1..?"


CARDINALITY_REQUIRED = [Cardinality.ONE, Cardinality.NONZERO]
CARDINALITY_SINGLE = [Cardinality.ONE, Cardinality.OPTIONAL]


class RelationshipDescription(NamedTuple):
    """
    Named tuple to describe the nature of the relationship between two concepts
    """

    relationship_id: str
    cardinality: Cardinality
    target_class: str  # TODO: deprecate, replace usages with target_definition
    target_definition: ConceptDefinition | None = None  # todo: remove default
