"""
Contains base classes required to define RxNorm, RxNorm extension and source
concepts
"""

from __future__ import annotations

from abc import ABC, abstractmethod  # For DrugNode interface
from typing import (
    NamedTuple,  # For ConceptCodeVocab
    override,  # for typing
)

type ConceptIdentifier = ConceptId | ConceptCodeVocab


# Identifiers
class ConceptId(int):
    """
    Unique identifier for a concept in the OMOP vocabulary.

    This is just a subclass of int, meaning CPython will treat it as an int.
    """


class ConceptCodeVocab(NamedTuple):
    """
    Vocabulary and code pair for a concept in the OMOP vocabulary.
    """

    concept_code: str
    vocabulary_id: str

    @override
    def __str__(self):
        return f"{self.vocabulary_id}/{self.concept_code}"


class HierarchyNode[Id](ABC):
    """
    Base class for all nodes in the drug concept hierarchy (Ingredient, Complex
    and Pack).

    This class provides a common interface for all nodes in the hierarchy.
    """

    identifier: Id

    @abstractmethod
    def is_superclass_of(
        self, other: HierarchyNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        """
        Check if this node is a superclass of another node.

        Args:
            other: The node to check against.
            passed_hierarchy_checks: Whether the tested node had already passed
                the corresponding checks by the predecessors of this node. This
                is used to avoid redundant checks in the hierarchy.
        """
