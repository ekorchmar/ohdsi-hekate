"""
Implementation of the pack classes
"""

from __future__ import annotations

from dataclasses import dataclass  # For classes
from typing import NoReturn, override

import rx_model.drug_classes.atom as a
from rx_model.drug_classes.base import (
    ConceptIdentifier,
    HierarchyNode,
)  # For identifiers
from rx_model.drug_classes.generic import (
    PackEntry,  # For contents
    PackNode,  # For interface
)
from utils.enums import ConceptClassId
from utils.classes import SortedTuple  # For typing
from utils.exceptions import PackCreationError  # For PackEntry


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalPack[Id: ConceptIdentifier](PackNode[Id]):
    """
    Pack node for clinical packs.
    """

    identifier: Id
    entries: SortedTuple[PackEntry[Id]]

    def __post_init__(self) -> None:
        if len(self.entries) == 0:
            raise PackCreationError(f"Pack {self.identifier} has no entries.")

        for entry in self.entries:
            if (msg := entry.validate_entry()) is not None:
                raise PackCreationError(
                    f"Pack {self.identifier} has invalid entry: {msg}"
                )

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
    def is_superclass_of(
        self, other: HierarchyNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        # Only packs can be subsumed
        if not isinstance(other, PackNode):
            return False

        if not passed_hierarchy_checks:
            # Make sure that other node's pack entries match ours
            if len(other.get_entries()) != len(self.entries) or any(
                not p_entry.semantic_ancestor_of(n_entry)
                for n_entry, p_entry in zip(self.entries, other.get_entries())
            ):
                return False

        return True

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[PackNode[Id]]],
        attributes: dict[ConceptClassId, a.BrandName[Id] | a.Supplier[Id]],
        entries: SortedTuple[PackEntry[Id]],
    ) -> ClinicalPack[Id]:
        return cls(identifier=identifier, entries=entries)


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
    def is_superclass_of(
        self, other: HierarchyNode[Id], passed_hierarchy_checks: bool = True
    ) -> NoReturn:
        raise NotImplementedError

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[PackNode[Id]]],
        attributes: dict[ConceptClassId, a.BrandName[Id] | a.Supplier[Id]],
        entries: SortedTuple[PackEntry[Id]],
    ) -> NoReturn:
        raise NotImplementedError
