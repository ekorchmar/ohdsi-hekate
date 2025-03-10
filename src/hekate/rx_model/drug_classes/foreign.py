"""
Contains the ForeignDrugNode class, which represents an unknown node in the
drug concept hierarchy.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn, override, NamedTuple
from rx_model.drug_classes.generic import (
    ConceptIdentifier,
    BoundStrength,
    DrugNode,
)
import rx_model.drug_classes.atom as a
import rx_model.drug_classes.strength as st
import rx_model.drug_classes.complex as c

from utils.classes import SortedTuple
from utils.exceptions import ForeignNodeCreationError
from utils.utils import count_repeated_first_entries

type _AnyComplex[Id: ConceptIdentifier] = (
    a.Ingredient[Id]  # Actually identifies possible multiple types
    | c.ClinicalDrugComponent[Id, st.UnquantifiedStrength]  # ditto
    | c.BrandedDrugComponent[Id, st.UnquantifiedStrength]
    | c.ClinicalDrugForm[Id]
    | c.BrandedDrugForm[Id]
    | c.ClinicalDrug[Id, st.UnquantifiedStrength]
    | c.BrandedDrug[Id, st.UnquantifiedStrength]
    | c.QuantifiedClinicalDrug[Id, c.Concentration]
    | c.QuantifiedBrandedDrug[Id]
)

# PseudoUnit is verbatim string representation of a unit in source data
type PseudoUnit = str


@dataclass(frozen=True, slots=True)
class ForeignDrugNode[Id: ConceptIdentifier, S: st.Strength | None](
    DrugNode[Id, S]
):
    """
    Represents an unknown node in the drug concept hierarchy. This is used to
    represent source drug concepts that may not be present in the RxHierarchy,
    """

    identifier: Id
    strength_data: SortedTuple[BoundStrength[Id, S]]
    brand_name: a.BrandName[Id] | None = None
    dose_form: a.DoseForm[Id] | None = None
    supplier: a.Supplier[Id] | None = None

    precise_ingredients: Sequence[a.PreciseIngredient | None] | None = None

    @override
    def is_superclass_of(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> NoReturn:
        del passed_hierarchy_checks, other
        raise NotImplementedError(
            "Cannot check superclass relationship with a foreign node."
        )

    @override
    def get_strength_data(
        self,
    ) -> SortedTuple[BoundStrength[Id, S]]:
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

    def __post_init__(self):
        self.validate_strength_data()
        self.validate_precise_ingredients()
        self.forbid_formless_quantities()
        # TODO: Marketed Product checks

    def validate_strength_data(self):
        """
        Ensures that the strength data is valid.
        """
        if not self.strength_data:
            raise ForeignNodeCreationError(
                "Foreign nodes must have at least one strength data entry, but "
                f"Node {self.identifier} has none."
            )

        if len(self.strength_data) == 1:
            # Nothing to validate
            return

        # Strength or None, this set must contain exactly one type
        strength_types = {type(strength) for _, strength in self.strength_data}
        if len(strength_types) != 1:
            raise ForeignNodeCreationError(
                "All strength data must be of the same type, but Node "
                f"{self.identifier} has: {strength_types}."
            )

        # Can not have more than one ingredient
        repeated_ingredients = count_repeated_first_entries(self.strength_data)
        if repeated_ingredients:
            repeats = ", ".join(
                f"{v} of {k}" for k, v in repeated_ingredients.items()
            )
            raise ForeignNodeCreationError(
                "All strength data must have unique ingredients, but Node "
                f"{self.identifier} has: {repeats}."
            )

        # If the strength data is quantified, ensure that denominators are
        # the same
        first, *others = self.strength_data
        if isinstance(first[1], st.LiquidQuantity):
            for other in others:
                if first[1].denominator_matches(other[1]):
                    continue
                raise ForeignNodeCreationError(
                    "All strength data must have the same denominator, but "
                    f"Node {self.identifier} has: {first[1].denominator_value} "
                    f"{first[1].denominator_unit} and "
                    f"{other[1].denominator_value} and "
                    f"{other[1].denominator_unit}."
                )

    def validate_precise_ingredients(self):
        """
        Ensures that the precise ingredients are valid.
        """
        if self.precise_ingredients is None:
            return

        if len(self.precise_ingredients) != len(self.strength_data):
            raise ForeignNodeCreationError(
                f"If precise ingredients are provided, there must be one for "
                f"each strength data entry, or explicitly set to None; but "
                f"Node has: {len(self.precise_ingredients)} for "
                f"{len(self.strength_data)}."
            )

        for (ing, _), pi in zip(self.strength_data, self.precise_ingredients):
            if pi is None:
                continue
            if pi.invariant != ing:
                raise ForeignNodeCreationError(
                    f"Precise ingredient {pi} corresponds to ingredient {ing} "
                    f"in Node {self.identifier}, but it is not a known variant."
                )

    def forbid_formless_quantities(self):
        """
        Ensures that all strength data entries have a dose form.
        """
        # TODO: this check is optional; run parameter should be added to
        # treat it as a warning
        if self.dose_form is not None:
            return

        for _, strength in self.strength_data:
            if isinstance(strength, st.LiquidQuantity):
                raise ForeignNodeCreationError(
                    f"All quantified strength data entries must have a dose "
                    f"form, but Node {self.identifier} does not have one."
                )

    def best_case_class(self) -> type[_AnyComplex[Id]]:
        """
        Tries to infer the target class of this foreign node based on the
        presence of attributes and shape of the strength data.
        """

        branded = self.brand_name is not None
        marketed = self.supplier is not None
        with_form = self.dose_form is not None
        _, strength = self.strength_data[0]

        # TODO: Marketed Product checks
        del marketed

        if strength is None:
            # Ingredient, CDF, or BDF
            if with_form:
                return c.BrandedDrugForm if branded else c.ClinicalDrugForm
            return a.Ingredient
        elif isinstance(strength, st.LiquidQuantity):
            # QCD or QBD
            return (
                c.QuantifiedBrandedDrug if branded else c.QuantifiedClinicalDrug
            )
        elif with_form:
            # CD or BD
            return c.BrandedDrug if branded else c.ClinicalDrug
        else:  # Unquantified strength, no form
            # CDC or BDC
            return (
                c.BrandedDrugComponent if branded else c.ClinicalDrugComponent
            )

    def is_multi(self) -> bool:
        """
        Returns True if this node contains multiple ingredients or strength
        entries.
        """
        return len(self.strength_data) > 1


class ForeignStrength(NamedTuple):
    """
    Represents a strength entry in a foreign node.

    It is purposefully detached from the ingredient information, as the logic
    for handling mapping information is processed separately.
    """

    amount_value: float | None
    amount_unit: PseudoUnit | None
    numerator_value: float | None
    numerator_unit: PseudoUnit | None
    denominator_value: float | None
    denominator_unit: PseudoUnit | None
    box_size: int | None
