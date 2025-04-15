"""
Contains generic high level supertypes and interfaces for the drug
concept hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override  # for HierarchyNode interface
from dataclasses import dataclass  # for PackEntry

import rx_model.drug_classes.atom as a
import rx_model.drug_classes.strength as st
from rx_model.drug_classes.base import (
    ConceptIdentifier,  # For identifiers
    HierarchyNode,  # For node interface
)
from utils.classes import SortedTuple  # For typing
from utils.enums import ConceptClassId  # For typing
from utils.constants import BOX_SIZE_LIMIT  # aka Postgres smallint limit


class DrugNode[
    Id: ConceptIdentifier,
    S: st.Strength | None,
](HierarchyNode[Id], ABC):
    """
    Metaclass for the nodes in the drug concept hierarchy.

    Purpose of this class is to provide a consistent interface for the
    transitive closure methods, allowing dynamic typing to be used.
    """

    identifier: Id

    @abstractmethod
    def get_strength_data(
        self,
    ) -> SortedTuple[st.BoundStrength[Id, S]]:
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
        strength_data: SortedTuple[st.BoundStrength[Id, S]],
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

    @abstractmethod
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        """
        Check if this node is a superclass of another DrugNode.

        Args:
            other: The node to check against.
            passed_hierarchy_checks: Whether the tested node had already passed
                the corresponding checks by the predecessors of this node. This
                is used to avoid redundant checks in the hierarchy.
        """

    @override
    def is_superclass_of(
        self, other: HierarchyNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        """
        Check if this node is a superclass of another HierarchyNode.
        """
        match other:
            case DrugNode():
                return self.is_superclass_of_drug_node(
                    other,  # pyright: ignore[reportUnknownArgumentType]  # noqa: E501
                    passed_hierarchy_checks,
                )
            case PackNode():
                raise NotImplementedError(
                    "Testing if drug node is superclass of pack node is not "
                    "yet implemented."
                )
            case a.Ingredient():
                # NOTE: A bit weird we did end up here. Raising an error is
                # better than returning False, although would make sense

                raise TypeError(
                    "DrugNode should not be tested as Ingredient superclass."
                )
                # return False
            case _:
                # Unreachable
                raise ValueError(f"{other} is not a valid node type to test.")


@dataclass(frozen=True, eq=True, slots=True)
class PackEntry[Id: ConceptIdentifier]:
    """
    Drug entry in a pack
    """

    drug: DrugNode[Id, st.LiquidQuantity | st.SolidStrength]
    amount: int | None
    box_size: int | None

    def __lt__(self, other: PackEntry[Id]) -> bool:
        """
        Enable sorting of pack entries
        """
        # NOTE: Comparison between different identifier types can only occur in
        # a programming error state
        return self.drug.identifier < other.drug.identifier  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]  # noqa: 501

    def validate_entry(self) -> str | None:
        """
        Validate the pack entry. Returns either a string with the error message
        or None if the entry is valid.
        """
        strength_data = self.drug.get_strength_data()
        strength_instance: st.Strength = strength_data[0][1]
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            strength_instance, (st.LiquidQuantity, st.SolidStrength)
        ):
            return (
                f"Pack entry {self.drug.identifier} "
                f"{self.drug.__class__.__name__} has unquantified strength and "
                f"can not participate in pack creation: {strength_data}"
            )

        if self.box_size is not None and 0 >= self.box_size >= BOX_SIZE_LIMIT:
            return (
                f"Box size {self.box_size} is not valid. Must be between"
                f"0 and {BOX_SIZE_LIMIT}"
            )
        if self.amount is not None and 0 >= self.amount >= BOX_SIZE_LIMIT:
            return (
                f"Box size {self.amount} is not valid. Must be between"
                f"0 and {BOX_SIZE_LIMIT}"
            )

    def semantic_ancestor_of(self, other: PackEntry[Id]) -> bool:
        return (
            # Entries are expected to match drug on identifier
            self.drug.identifier == other.drug.identifier
            and (self.amount or 1) == (other.amount or 1)
            and ((self.box_size is None) or self.box_size == other.box_size)
        )


class PackNode[Id: ConceptIdentifier](HierarchyNode[Id], ABC):
    """
    Metaclass for the pack nodes in the drug concept hierarchy.

    Purpose of this class is to provide a consistent interface for the
    transitive closure methods, allowing dynamic typing to be used.
    """

    identifier: Id

    @abstractmethod
    def get_entries(self) -> SortedTuple[PackEntry[Id]]:
        """
        Get the entries of the pack.
        """

    @abstractmethod
    def get_brand_name(self) -> a.BrandName[Id] | None:
        """
        Get the brand name of the pack.
        """

    @abstractmethod
    def get_supplier(self) -> a.Supplier[Id] | None:
        """
        Get the supplier of the pack.
        """

    @classmethod
    @abstractmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[PackNode[Id]]],
        attributes: dict[ConceptClassId, a.BrandName[Id] | a.Supplier[Id]],
        entries: SortedTuple[PackEntry[Id]],
    ) -> PackNode[Id]:
        """
        Create a pack node from the definitions.
        """
