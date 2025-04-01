"""
Implementation of the pack classes
"""

from __future__ import annotations

from abc import ABC, abstractmethod  # For PackNode interface
from dataclasses import dataclass  # For classes
from typing import NoReturn, override

from hekate.utils.enums import ConceptClassId
import rx_model.drug_classes.atom as a
import rx_model.drug_classes.strength as st
from rx_model.drug_classes.generic import (
    ConceptIdentifier,  # For identifiers
    DrugNode,  # For contents
)
from utils.classes import SortedTuple  # For typing
from utils.constants import BOX_SIZE_LIMIT  # aka Postgres smallint limit
from utils.exceptions import PackCreationError  # For PackEntry


@dataclass(frozen=True, order=True, eq=True, slots=True)
class PackEntry[Id: ConceptIdentifier]:
    """
    Drug entry in a pack
    """

    drug: DrugNode[Id, st.LiquidQuantity | st.SolidStrength]
    amount: int | None
    box_size: int | None

    def __post_init__(self):
        # TODO: validate the drug (e.g. no box size, quantified)

        if self.box_size is not None and 0 >= self.box_size >= BOX_SIZE_LIMIT:
            raise PackCreationError(
                f"Box size {self.box_size} is not valid. Must be between"
                f"0 and {BOX_SIZE_LIMIT}"
            )
        if self.amount is not None and 0 >= self.amount >= BOX_SIZE_LIMIT:
            raise PackCreationError(
                f"Box size {self.amount} is not valid. Must be between"
                f"0 and {BOX_SIZE_LIMIT}"
            )


class PackNode[Id: ConceptIdentifier](ABC):
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
    def get_supplier(self) -> a.BrandName[Id] | None:
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
        entries: list[PackEntry[Id]],
    ) -> PackNode[Id]:
        """
        Create a pack node from the definitions.
        """


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalPack[Id: ConceptIdentifier](PackNode[Id]):
    """
    Pack node for clinical packs.
    """

    identifier: Id
    entries: SortedTuple[PackEntry[Id]]

    def __post_init__(self) -> NoReturn:
        # TODO: implement checks
        raise NotImplementedError("Clinical packs are not implemented yet.")

    @override
    def get_brand_name(self) -> None:
        return None

    @override
    def get_supplier(self) -> None:
        return None

    @override
    def get_entries(self) -> SortedTuple[PackEntry[Id]]:
        return self.entries

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[PackNode[Id]]],
        attributes: dict[ConceptClassId, a.BrandName[Id] | a.Supplier[Id]],
        entries: list[PackEntry[Id]],
    ) -> NoReturn:
        raise NotImplementedError


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BrandedPack[Id: ConceptIdentifier](PackNode[Id]):
    """
    Pack node for clinical packs.
    """

    identifier: Id
    unbranded: ClinicalPack[Id]
    brand_name: a.BrandName[Id]

    def __post_init__(self) -> NoReturn:
        # TODO: implement checks
        raise NotImplementedError("Branded packs are not implemented yet.")

    @override
    def get_entries(self) -> SortedTuple[PackEntry[Id]]:
        return self.unbranded.get_entries()

    @override
    def get_brand_name(self) -> a.BrandName[Id]:
        return self.brand_name

    @override
    def get_supplier(self) -> None:
        return None

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[PackNode[Id]]],
        attributes: dict[ConceptClassId, a.BrandName[Id] | a.Supplier[Id]],
        entries: list[PackEntry[Id]],
    ) -> NoReturn:
        raise NotImplementedError
