"""
Contains the ForeignDrugNode class, which represents an unknown node in the
drug concept hierarchy.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn, override
from rx_model.drug_classes.generic import (
    ConceptIdentifier,
    BoundStrength,
    DrugNode,
)
import rx_model.drug_classes.atom as a
import rx_model.drug_classes.strength as st
import rx_model.drug_classes.complex as c

from utils.classes import SortedTuple

type _AnyComplex[Id: ConceptIdentifier] = (
    SortedTuple[a.Ingredient[Id]]
    | c.ClinicalDrugComponent[Id, st.UnquantifiedStrength]
    | c.BrandedDrugComponent[Id, st.UnquantifiedStrength]
    | c.ClinicalDrugForm[Id]
    | c.BrandedDrugForm[Id]
    | c.ClinicalDrug[Id, st.UnquantifiedStrength]
    | c.BrandedDrug[Id, st.UnquantifiedStrength]
    | c.QuantifiedClinicalDrug[Id, c.Concentration]
    | c.QuantifiedBrandedDrug[Id, c.Concentration]
)


@dataclass(frozen=True, slots=True)
class ForeignDrugNode[Id: ConceptIdentifier](DrugNode[Id]):
    """
    Represents an unknown node in the drug concept hierarchy. This is used to
    represent source drug concepts that may not be present in the RxHierarchy,
    """

    identifier: Id
    strength_data: SortedTuple[BoundStrength[Id, st.Strength | None]]
    brand_name: a.BrandName[Id] | None = None
    dose_form: a.DoseForm[Id] | None = None
    supplier: a.Supplier[Id] | None = None

    precise_ingredients: Sequence[a.PreciseIngredient | None] | None = None

    @override
    def is_superclass_of(
        self,
        other: DrugNode[Id],
        passed_hierarchy_checks: bool = True,
    ) -> NoReturn:
        raise NotImplementedError(
            "Cannot check superclass relationship with a foreign node."
        )

    @override
    def get_strength_data(
        self,
    ) -> SortedTuple[BoundStrength[Id, st.Strength | None]]:
        return self.strength_data

    @override
    def get_precise_ingredients(
        self,
    ) -> Sequence[a.PreciseIngredient | None]:
        # If no precise ingredients are provided, return a list of Nones
        if self.precise_ingredients is None:
            return [None] * len(self.strength_data)
        return self.precise_ingredients

    @override
    def get_brand_name(self) -> a.BrandName[Id] | None:
        return self.brand_name

    @override
    def get_dose_form(self) -> a.DoseForm[Id] | None:
        return self.dose_form

    @override
    def get_supplier(self) -> a.Supplier[Id] | None:
        return self.supplier

    def target_class(self) -> _AnyComplex[Id]:
        """
        Tries to infer the target class of this foreign node based on the
        presence of attributes and shape of the strength data.
        """
        raise NotImplementedError("Not yet implemented.")
