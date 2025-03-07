"""
Contains generic data classes and types used throughout the project.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from utils.classes import SortedTuple

if TYPE_CHECKING:
    import rx_model.drug_classes.atom as a
    import rx_model.drug_classes.strength as st


# Identifiers
class ConceptId(int):
    """
    Unique identifier for a concept in the OMOP vocabulary.

    This is just a subclass of int, meaning CPython will treat it as an int.
    """


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ConceptCodeVocab:
    """
    Vocabulary and code pair for a concept in the OMOP vocabulary.
    """

    vocabulary_id: str
    concept_code: str

    @override
    def __str__(self):
        return f"{self.vocabulary_id}:{self.concept_code}"


type ConceptIdentifier = ConceptId | ConceptCodeVocab

type BoundStrength[Id: ConceptIdentifier, S: "st.Strength | None"] = tuple[
    "a.Ingredient[Id]", S
]


class DrugNode[Id: ConceptIdentifier, S: "st.Strength | None"](ABC):
    """
    Metaclass for the nodes in the drug concept hierarchy.

    Purpose of this class is to provide a consistent interface for the
    transitive closure methods, allowing dynamic typing to be used.
    """

    identifier: Id

    @abstractmethod
    def is_superclass_of(
        self,
        other: "DrugNode[Id, st.Strength | None]",
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
    ) -> SortedTuple[BoundStrength[Id, "st.Strength | None"]]:
        """
        Retrieve all strength data for this node. Every entry will always
        specify an ingredient and a strength, which may be None for nodes
        that do not have strength data (e.g. ingredients).
        """

    @abstractmethod
    def get_precise_ingredients(self) -> Sequence["a.PreciseIngredient | None"]:
        """
        Retrieve all Precise a.Ingredients participating in this node. Data
        will be returned as a sequence of Precise Ingredient instances or None,
        matching layout with the `get_strength_data` method.
        """

    # Methods to get a possibly inherited attribute
    # NOTE: These methods default to returning None. Be careful in subclasses!
    def get_brand_name(self) -> "a.BrandName[Id] | None":
        """
        Retrieve the brand name for this node.
        """
        return None

    def get_dose_form(self) -> "a.DoseForm[Id] | None":
        """
        Retrieve the dose form for this node.
        """
        return None

    def get_supplier(self) -> "a.Supplier[Id] | None":
        """
        Retrieve the a.supplier for this node.
        """
        return None
