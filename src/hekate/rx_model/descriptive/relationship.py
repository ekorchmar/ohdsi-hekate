"""
Contains OMOP representation of intra- RxNorm and RxNorm Extension
relationships.
"""

from typing import NamedTuple
from rx_model.descriptive.base import ConceptDefinition
from utils.enums import Cardinality


CARDINALITY_REQUIRED = [Cardinality.ONE, Cardinality.NONZERO]
CARDINALITY_SINGLE = [Cardinality.ONE, Cardinality.OPTIONAL]


class RelationshipDescription(NamedTuple):
    """
    Named tuple to describe the nature of the relationship between two concepts
    """

    relationship_id: str
    cardinality: Cardinality
    target_definition: ConceptDefinition
