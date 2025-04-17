"""
Implementation of the pack classes
"""

from __future__ import annotations

from dataclasses import dataclass  # For classes
from abc import ABC  # For metaclasses
from typing import ClassVar, override

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


class __MetaClinicalPack[Id: ConceptIdentifier](PackNode[Id], ABC):
    """
    Metaclass containing shared logic for Clinical Packs and Clinical Pack
    Box
    """

    identifier: Id
    entries: SortedTuple[PackEntry[Id]]

    defines_box_size: ClassVar[bool]

    def __post_init__(self) -> None:
        if len(self.entries) == 0:
            raise PackCreationError(f"Pack {self.identifier} has no entries.")

        for entry in self.entries:
            if (entry.box_size is not None) != self.defines_box_size:
                raise PackCreationError(
                    f"{self.__class__.__name__} {self.identifier} has entries "
                    + ("omitting" if self.defines_box_size else "specifying")
                    + " box_size, which is not allowed for this class"
                )

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


class __MetaBrandedPack[Id: ConceptIdentifier](PackNode[Id], ABC):
    """
    Metaclass containing shared logic for Branded Packs and Branded Pack Box
    """

    identifier: Id
    unbranded: __MetaClinicalPack[Id]
    brand_name: a.BrandName[Id]

    def __post_init__(self) -> None:
        # NOTE: although current branded pack contents are automatically
        # reduced to clinical counterparts, this may change in the future
        for i, entry in enumerate(self.get_entries()):
            if entry_brand_name := entry.drug.get_brand_name():
                if not entry_brand_name == self.get_brand_name():
                    raise PackCreationError(
                        f"Entry #{i} in {self.__class__.__name__} "
                        f"{self.identifier} specifies {entry_brand_name}, but "
                        f"the pack has {self.get_brand_name()}"
                    )

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
    ) -> bool:
        # Only packs can be subsumed
        if not isinstance(other, PackNode):
            return False

        if not passed_hierarchy_checks and not self.unbranded.is_superclass_of(
            other
        ):
            return False

        return self.get_brand_name() == other.get_brand_name()

    # NOTE: from_definitions alternative constructor is not overriden


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalPack[Id: ConceptIdentifier](__MetaClinicalPack[Id]):
    """
    Pack node for clinical packs.
    """

    identifier: Id
    entries: SortedTuple[PackEntry[Id]]

    defines_box_size: ClassVar[bool] = False

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[PackNode[Id]]],
        attributes: dict[ConceptClassId, a.BrandName[Id] | a.Supplier[Id]],
        entries: SortedTuple[PackEntry[Id]],
    ) -> __MetaClinicalPack[Id]:
        return cls(identifier=identifier, entries=entries)


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BrandedPack[Id: ConceptIdentifier](__MetaBrandedPack[Id]):
    """
    Pack node for clinical packs.
    """

    identifier: Id
    unbranded: ClinicalPack[Id]  # pyright: ignore[reportIncompatibleVariableOverride]  # noqa: E501
    brand_name: a.BrandName[Id]

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[PackNode[Id]]],
        attributes: dict[ConceptClassId, a.BrandName[Id] | a.Supplier[Id]],
        entries: SortedTuple[PackEntry[Id]],
    ) -> BrandedPack[Id]:
        (cp_node,) = parents[ConceptClassId.CP]
        if not isinstance(cp_node, ClinicalPack):
            raise PackCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"ClinicalPack, but has {cp_node}."
            )
        brand_name = attributes[ConceptClassId.BRAND_NAME]
        assert isinstance(brand_name, a.BrandName)
        return cls(
            identifier=identifier, unbranded=cp_node, brand_name=brand_name
        )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalPackBox[Id: ConceptIdentifier](__MetaClinicalPack[Id]):
    """
    Pack node for clinical packs extended with box size.
    """

    identifier: Id
    entries: SortedTuple[PackEntry[Id]]
    unboxed: ClinicalPack[Id]

    defines_box_size: ClassVar[bool] = True

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[PackNode[Id]]],
        attributes: dict[ConceptClassId, a.BrandName[Id] | a.Supplier[Id]],
        entries: SortedTuple[PackEntry[Id]],
    ) -> ClinicalPackBox[Id]:
        (cp_node,) = parents[ConceptClassId.CP]
        if not isinstance(cp_node, ClinicalPack):
            raise PackCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"ClinicalPack, but has {cp_node}."
            )
        return cls(
            identifier=identifier,
            entries=entries,
            unboxed=cp_node,
        )
