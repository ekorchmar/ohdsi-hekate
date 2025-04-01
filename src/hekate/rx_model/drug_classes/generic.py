"""
Contains generic data classes and types used throughout the project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod  # For DrugNode interface
from typing import (
    TYPE_CHECKING,  # For conditional imports for typechecking
    NamedTuple,  # For ConceptCodeVocab
    override,  # for typing
)

from utils.classes import SortedTuple  # For typing
from utils.enums import ConceptClassId  # For typing

if TYPE_CHECKING:
    # Circular import: atom.Ingredient is also a DrugNode
    import rx_model.drug_classes.atom as a

    # Circular import: Strength needs ConceptId
    import rx_model.drug_classes.strength as st


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


type ConceptIdentifier = ConceptId | ConceptCodeVocab

type BoundStrength[Id: ConceptIdentifier, S: st.Strength | None] = tuple[
    "a.Ingredient[Id]", S
]


class DrugNode[Id: ConceptIdentifier, S: st.Strength | None](ABC):
    """
    Metaclass for the nodes in the drug concept hierarchy.

    Purpose of this class is to provide a consistent interface for the
    transitive closure methods, allowing dynamic typing to be used.
    """

    identifier: Id

    @abstractmethod
    def is_superclass_of(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        """
        Check if this node is a superclass of another node.

        Args:
            other: The node to check against.
            passed_hierarchy_checks: Whether the tested node had already passed
                the corresponding checks by the predecessors of this node. This
                is used to avoid redundant checks in the hierarchy.
        """

    @abstractmethod
    def get_strength_data(
        self,
    ) -> SortedTuple[BoundStrength[Id, S]]:
        """
        Retrieve all strength data for this node. Every entry will always
        specify an ingredient and a strength, which may be None for nodes
        that do not have strength data (e.g. ingredients).
        """

    @abstractmethod
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        """
        Retrieve all Precise a.Ingredients participating in this node. Data
        will be returned as a sequence of Precise Ingredient instances or None,
        matching layout with the `get_strength_data` method.
        """

    # Methods to get a possibly inherited attribute
    # NOTE: These methods default to returning None. Be careful in subclasses!
    def get_brand_name(self) -> a.BrandName[Id] | None:
        """
        Retrieve the brand name for this node.
        """
        return None

    def get_dose_form(self) -> a.DoseForm[Id] | None:
        """
        Retrieve the dose form for this node.
        """
        return None

    def get_supplier(self) -> a.Supplier[Id] | None:
        """
        Retrieve the a.supplier for this node.
        """
        return None

    def get_box_size(self) -> int | None:
        """
        Retrieve the box size for this node.
        """
        return None

    @classmethod
    @abstractmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[DrugNode[Id, st.Strength | None]]],
        attributes: dict[
            ConceptClassId, a.BrandName[Id] | a.DoseForm[Id] | a.Supplier[Id]
        ],
        precise_ingredients: list[a.PreciseIngredient],
        strength_data: SortedTuple[BoundStrength[Id, S]],
        box_size: int | None,
    ) -> DrugNode[Id, S]:
        """
        Create a DrugNode instance from concept definitions.

        Note that definitions are allowed to be redundant, so each class
        implments its own logic to pick and choose the correct data.
        """
        raise NotImplementedError(
            f"{cls.__name__} should not be constructed from definitions."
        )
