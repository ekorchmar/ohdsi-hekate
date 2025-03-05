from dataclasses import dataclass
from typing import override

from utils.classes import SortedTuple

from rx_model import exception
from rx_model.drug_classes.generic import (
    BoundStrength,
    ConceptId,
    ConceptIdentifier,
    DrugNode,
)


# Atomic named concepts
@dataclass(frozen=True, order=True, slots=True)
class __RxAtom[Id: ConceptIdentifier]:
    """A single atomic concept in the RxNorm vocabulary."""

    identifier: Id
    concept_name: str

    def __post_init__(self):
        if not self.concept_name:
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} {self.identifier}: name must not "
                f"be empty."
            )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.identifier == other.identifier


# RxNorm
class Ingredient[Id: ConceptIdentifier](__RxAtom[Id], DrugNode[Id]):
    """
    RxNorm or RxNorm Extension ingredient concept.
    """

    @override
    def is_superclass_of(
        self, other: DrugNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        del passed_hierarchy_checks
        # Ingredients are superclasses of any node containing them
        return any(self == ing for ing, _ in other.get_strength_data())

    @override
    def get_strength_data(self) -> SortedTuple[BoundStrength[Id, None]]:
        return SortedTuple([(self, None)])

    @override
    def get_precise_ingredients(self) -> list[None]:
        return [None]


class BrandName[Id: ConceptIdentifier](__RxAtom[Id]):
    """
    RxNorm or RxNorm Extension brand name concept.
    """


class DoseForm[Id: ConceptIdentifier](__RxAtom[Id]):
    """
    RxNorm or RxNorm Extension dose form concept.
    """


# # UCUM
class Unit(__RxAtom[ConceptId]):
    """
    UCUM unit concept used in drug dosage information.
    """


@dataclass(frozen=True, order=True, eq=True, slots=True)
class PreciseIngredient(__RxAtom[ConceptId]):
    invariant: Ingredient[ConceptId]


# # RxNorm Extension
class Supplier[Id: ConceptIdentifier](__RxAtom[Id]):
    """
    RxNorm Extension supplier concept.
    """
