from __future__ import annotations

from dataclasses import dataclass  # For atomic concepts
from typing import ClassVar, NoReturn, final, override

from rx_model.drug_classes.base import (
    ConceptId,
    ConceptIdentifier,
    HierarchyNode,
)
from utils.exceptions import RxConceptCreationError  # For error handling


# Atomic named concepts
@dataclass(frozen=True)
class _RxAtom[Id: ConceptIdentifier]:
    """A single atomic concept in the RxNorm vocabulary."""

    identifier: Id
    concept_name: str

    __slots__: ClassVar[tuple[str, ...]] = ("identifier", "concept_name")

    def __post_init__(self):
        if not self.concept_name:
            raise RxConceptCreationError(
                f"{self.__class__.__name__} {self.identifier}: name must not "
                f"be empty."
            )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            # This is a common case, False is intended
            if other is None:
                return False

            raise TypeError(
                f"Cannot compare {self.__class__.__name__} with "
                f"{other.__class__.__name__}."
            )
        return other is self or self.identifier == other.identifier

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, _RxAtom):
            return NotImplemented
        return self.identifier > other.identifier  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]  # noqa: E501

    @override
    def __hash__(self) -> int:
        return hash((self.identifier, self.__class__.__name__))


# RxNorm
@final
class Ingredient[Id: ConceptIdentifier](_RxAtom[Id], HierarchyNode[Id]):
    """
    RxNorm or RxNorm Extension ingredient concept.
    """

    @override
    def is_superclass_of(
        self,
        other: HierarchyNode[Id],
        passed_hierarchy_checks: bool = True,
    ) -> NoReturn:
        raise NotImplementedError(
            "Ingredients are meant to be entry points to the search space, "
            "known in advance. Generic interface for superclass testing should "
            "not be used."
        )


@final
class BrandName[Id: ConceptIdentifier](_RxAtom[Id]):
    """
    RxNorm or RxNorm Extension brand name concept.
    """


@final
class DoseForm[Id: ConceptIdentifier](_RxAtom[Id]):
    """
    RxNorm or RxNorm Extension dose form concept.
    """


@final
@dataclass(frozen=True, unsafe_hash=True)
class PreciseIngredient(_RxAtom[ConceptId]):
    __slots__: ClassVar[tuple[str, ...]] = (
        "identifier",
        "concept_name",
        "invariant",
    )
    invariant: Ingredient[ConceptId]

    # NOTE: Since @dataclass overrides a lot of superclass methods, (and we need
    # to call it to redefine the constructor) we need to explicitly reimplement
    # dunder methods.

    @override
    def __eq__(self, other: object) -> bool:
        # Precise Ingredients are expected to be compared to None
        # more often than not, so we elevate this to an outer check
        if other is None:
            return False

        # Shorthand for another common case
        if other is self:
            return True

        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot compare {self.__class__.__name__} with "
                f"{other.__class__.__name__}."
            )
        return self.identifier == other.identifier

    @override
    def __gt__(self, other: object) -> bool:
        if not isinstance(other, PreciseIngredient):
            return NotImplemented
        return self.identifier > other.identifier


# # UCUM
@final
class Unit(_RxAtom[ConceptId]):
    """
    UCUM unit concept used in drug dosage information.
    """


# # RxNorm Extension
@final
class Supplier[Id: ConceptIdentifier](_RxAtom[Id]):
    """
    RxNorm Extension supplier concept.
    """
