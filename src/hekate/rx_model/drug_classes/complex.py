from dataclasses import dataclass
from typing import override  # For type annotations

import rx_model.drug_classes.strength as st
from rx_model import exception  # For custom exceptions
from rx_model.drug_classes import atom as a
from rx_model.drug_classes.generic import (
    BoundStrength,
    ConceptIdentifier,
    DrugNode,
)
from utils.classes import SortedTuple  # To ensure consistent layout
from utils.utils import invert_merge_dict, keep_multiple_values


class __MulticomponentMixin[
    Id: ConceptIdentifier,
    S: "st.UnquantifiedStrength",
]:
    """Mixin for classes to implement checks with multiple components."""

    identifier: Id

    def check_multiple_components(
        self, container: SortedTuple["ClinicalDrugComponent[Id, S]"]
    ) -> None:
        if len(container) == 0:
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} {self.identifier} must "
                f"have at least one component."
            )

        # Check for duplicate ingredients
        duplicate_ingredients: dict[
            ClinicalDrugComponent[Id, S],
            "a.Ingredient[Id] | a.PreciseIngredient",
        ] = keep_multiple_values({
            cdc: cdc.precise_ingredient or cdc.ingredient for cdc in container
        })

        if duplicate_ingredients:
            msg = (
                f"{self.__class__.__name__} {self.identifier} "
                f"contains duplicate ingredients:"
            )
            for ing, cdcs in invert_merge_dict(duplicate_ingredients).items():
                msg += f" {ing.identifier} {ing.concept_name}"
                msg += f" ({ing.__class__.__name__}), coming from: "
                msg += " and ".join(str(cdc.identifier) for cdc in cdcs)
                msg += ";"
            raise exception.RxConceptCreationError(msg)


# Derived concepts
# # RxNorm
@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrugComponent[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    DrugNode[Id]
):
    """
    Single component containing (precise) ingredient and unquantified strength.
    """

    identifier: Id
    ingredient: a.Ingredient[Id]
    precise_ingredient: a.PreciseIngredient | None
    strength: S

    def __post_init__(self):
        if self.precise_ingredient is not None:
            pi, i = self.precise_ingredient, self.ingredient
            if pi.invariant != i:
                raise exception.RxConceptCreationError(
                    f"Error creating {self.__class__.__name__} with "
                    f"identifier {self.identifier}: stated precise ingredient "
                    f"{pi.identifier} {pi.concept_name} is not a variant of "
                    f"ingredient {i.identifier} {i.concept_name}."
                )

    @override
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        return [self.precise_ingredient]

    @override
    def is_superclass_of(
        self, other: DrugNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        del passed_hierarchy_checks
        # Components are superclasses of any node that contains this component

        # Match at least one strength
        for i, (ing, strength) in enumerate(other.get_strength_data()):
            # For performance reasons, test the ingredient first
            if ing != self.ingredient:
                continue

            # Can not match a node without strength data
            if strength is None:
                return False

            if self.strength.matches(strength):
                # If we specify a precise ingredient, it must match
                return (self.precise_ingredient is None) or (
                    self.precise_ingredient
                    == other.get_precise_ingredients()[i]
                )

        return False

    @override
    def get_strength_data(self) -> SortedTuple[BoundStrength[Id, S]]:
        return SortedTuple([(self.ingredient, self.strength)])


@dataclass(frozen=True, eq=True, slots=True)
class BrandedDrugComponent[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    __MulticomponentMixin[Id, S], DrugNode[Id]
):
    """\
Combination of clinical drug components with a stated brand name.

NB: Contains multiple components in one!\
"""

    identifier: Id
    clinical_drug_components: SortedTuple[ClinicalDrugComponent[Id, S]]
    brand_name: a.BrandName[Id]

    def __post_init__(self):
        self.check_multiple_components(self.clinical_drug_components)

    @override
    def get_strength_data(self) -> SortedTuple[BoundStrength[Id, S]]:
        # Return contents of unwrapped tuples
        return SortedTuple([
            cdc.get_strength_data()[0] for cdc in self.clinical_drug_components
        ])

    @override
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        return [cdc.precise_ingredient for cdc in self.clinical_drug_components]

    @override
    def is_superclass_of(
        self, other: DrugNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        if not passed_hierarchy_checks:
            if not all(
                cdc.is_superclass_of(other, passed_hierarchy_checks=False)
                for cdc in self.clinical_drug_components
            ):
                return False

        # Only check the brand name
        return self.brand_name == other.get_brand_name()

    @override
    def get_brand_name(self) -> a.BrandName[Id]:
        return self.brand_name


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrugForm[Id: ConceptIdentifier](DrugNode[Id]):
    identifier: Id
    dose_form: a.DoseForm[Id]
    ingredients: SortedTuple[a.Ingredient[Id]]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.dose_form

    def __post_init__(self):
        if len(self.ingredients) == 0:
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} must have at least one ingredient, "
                f"but {self.identifier} has none provided."
            )

        counts = {
            ingredient: self.ingredients.count(ingredient)
            for ingredient in self.ingredients
        }
        if any(count > 1 for count in counts.values()):
            msg = (
                f"{self.__class__.__name__} {self.identifier} contains "
                f"duplicate ingredients:"
            )
            for ing, cnt in counts.items():
                if cnt > 1:
                    msg += f" {ing.identifier} {ing.concept_name}"
                    msg += f"({cnt});"
            raise exception.RxConceptCreationError(msg)

    @override
    def is_superclass_of(
        self, other: DrugNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        if not passed_hierarchy_checks:
            # Check all ingredients
            if not all(ing.is_superclass_of(other) for ing in self.ingredients):
                return False

        # Only check the dose form
        return self.dose_form == other.get_dose_form()

    @override
    def get_strength_data(self) -> SortedTuple[BoundStrength[Id, None]]:
        return SortedTuple([(ing, None) for ing in self.ingredients])

    @override
    def get_precise_ingredients(self) -> list[None]:
        return [None] * len(self.ingredients)


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BrandedDrugForm[Id: ConceptIdentifier](DrugNode[Id]):
    identifier: Id
    clinical_drug_form: ClinicalDrugForm[Id]
    brand_name: a.BrandName[Id]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.clinical_drug_form.dose_form

    @override
    def get_brand_name(self) -> a.BrandName[Id]:
        return self.brand_name

    @override
    def is_superclass_of(
        self, other: DrugNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        if not passed_hierarchy_checks:
            if not self.clinical_drug_form.is_superclass_of(
                other, passed_hierarchy_checks=False
            ):
                return False

        # Only check the brand name
        return self.brand_name == other.get_brand_name()

    @override
    def get_strength_data(self) -> SortedTuple[BoundStrength[Id, None]]:
        return self.clinical_drug_form.get_strength_data()

    @override
    def get_precise_ingredients(self) -> list[None]:
        return self.clinical_drug_form.get_precise_ingredients()


# Prescriptable drug classes
# # Unquantified drugs
@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrug[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    __MulticomponentMixin[Id, S], DrugNode[Id]
):
    identifier: Id
    form: ClinicalDrugForm[Id]
    contents: SortedTuple[BoundStrength[Id, S]]
    # Those are the precise ingredients inherited from the components, if any
    precise_ingredients: tuple[a.PreciseIngredient | None]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.form.dose_form

    @override
    def get_precise_ingredients(self) -> tuple[a.PreciseIngredient | None]:
        return self.precise_ingredients

    @override
    def get_strength_data(self) -> SortedTuple[BoundStrength[Id, S]]:
        return self.contents

    @override
    def is_superclass_of(
        self, other: DrugNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        if not passed_hierarchy_checks:
            if not self.form.is_superclass_of(
                other, passed_hierarchy_checks=False
            ):
                return False

        # Check all components regardless of passed_hierarchy_checks
        # TODO: The fact that we arrived at this point means that the tested
        # node has passed the strength and content checks against the Comps.
        # However, current behavior of BuildRxE is to explicitly check the
        # components against the tested node. This is redundant, but we
        # implement it; but this entire block should be parametrized to not run
        # at all if passed_hierarchy_checks is True -- as a run parameter

        if len(self.contents) != len(other_data := other.get_strength_data()):
            return False

        shared_iter = enumerate(
            zip(
                self.contents,
                other_data,
                self.precise_ingredients,
            )
        )

        # NOTE: We assume that the components are in the same order as the
        # ingredients in the form thanks to the SortedTuple

        for i, (cd_data, node_data, pi) in shared_iter:
            cd_ing, cd_strength = cd_data
            node_ing, node_strength = node_data

            if node_strength is None:
                # Should be impossible, since we have already checked at least
                # one CDC
                return False

            if cd_ing != node_ing:
                return False

            if cd_strength.matches(node_strength):
                # Check PI
                if pi is not None or pi != other.get_precise_ingredients()[i]:
                    return False

        return True


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BrandedDrug[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    DrugNode[Id]
):
    # TODO: Check if dosage may be different from CD counterpart
    identifier: Id
    clinical_drug: ClinicalDrug[Id, S]
    brand_name: a.BrandName[Id]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.clinical_drug.form.dose_form

    @override
    def get_brand_name(self) -> a.BrandName[Id]:
        return self.brand_name

    @override
    def is_superclass_of(
        self, other: DrugNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        if not passed_hierarchy_checks:
            # Takes care of strength and dose form checks
            if not self.clinical_drug.is_superclass_of(
                other, passed_hierarchy_checks=False
            ):
                return False

        return self.brand_name == other.get_brand_name()

    @override
    def get_strength_data(self) -> SortedTuple[BoundStrength[Id, S]]:
        return self.clinical_drug.get_strength_data()

    @override
    def get_precise_ingredients(self) -> tuple[a.PreciseIngredient | None]:
        return self.clinical_drug.get_precise_ingredients()


# Quantified liquid forms
type _Concentration = st.LiquidConcentration | st.GaseousPercentage


@dataclass(frozen=True, order=True, eq=True, slots=True)
class QuantifiedClinicalDrug[Id: ConceptIdentifier, C: _Concentration](
    DrugNode[Id,]
):
    identifier: Id
    contents: SortedTuple[BoundStrength[Id, st.LiquidQuantity]]
    unquantified: ClinicalDrug[Id, C]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.unquantified.form.dose_form

    @override
    def get_precise_ingredients(self) -> tuple[a.PreciseIngredient | None]:
        return self.unquantified.get_precise_ingredients()

    def __post_init__(self):
        # Check strength data consistency
        unique_denoms: set[tuple[float, a.Unit]] = set()
        for _, quantity in self.contents:
            unit: a.Unit = quantity.denominator_unit
            value: float = quantity.denominator_value
            unique_denoms.add((value, unit))

        if len(unique_denoms) > 1:
            raise exception.RxConceptCreationError(
                f"Quantified clinical drug {self.identifier} must have "
                f"consistent denominator units and values."
            )

        # Check consistency against the unquantified form
        if len(self.contents) != len(self.unquantified.contents):
            raise exception.RxConceptCreationError(
                f"Quantified clinical drug {self.identifier} must have the "
                f"same number of components as its unquantified counterpart."
            )

        shared_iter = zip(self.contents, self.unquantified.contents)
        for (ing, quantity), (u_ing, strength) in shared_iter:
            if ing != u_ing:
                raise exception.RxConceptCreationError(
                    f"Quantified clinical drug {self.identifier} must have "
                    f"same ingredients as its unquantified counterpart."
                )

            if not quantity.matches(strength):
                raise exception.RxConceptCreationError(
                    f"Quantified clinical drug {self.identifier} must have "
                    f"same strength data as its unquantified counterpart for "
                    f"ingredient {ing.identifier} {ing.concept_name}."
                )

    @override
    def get_strength_data(
        self,
    ) -> SortedTuple[BoundStrength[Id, st.LiquidQuantity]]:
        return self.contents

    @override
    def is_superclass_of(
        self, other: DrugNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        if not passed_hierarchy_checks:
            if not self.unquantified.is_superclass_of(
                other, passed_hierarchy_checks=False
            ):
                return False

        if not self.get_dose_form() == other.get_dose_form():
            return False

        # Check all components regardless of passed_hierarchy_checks, as we have
        # quantified data
        if len(self.contents) != len(other_data := other.get_strength_data()):
            return False

        # Test if the other node has the same denominator unit and value
        if not isinstance(other_data[0][1], st.LiquidQuantity):
            # NOTE: By all rules, none of others are Quantities
            return False

        own_v, own_u = (
            self.contents[0][1].denominator_value,
            self.contents[0][1].denominator_unit,
        )
        other_v, other_u = (
            other_data[0][1].denominator_value,
            other_data[0][1].denominator_unit,
        )
        diff = own_v / other_v
        if not (st.LOW <= diff <= st.HIGH) or own_u != other_u:
            return False

        shared_iter = zip(self.contents, other_data)
        for (ing, quantity), (o_ing, o_strength) in shared_iter:
            assert o_strength is not None

            if ing != o_ing:
                return False

            if not quantity.matches(o_strength):
                return False

            # Precise ingredients are checked in the unquantified form

        return True


@dataclass(frozen=True, order=True, eq=True, slots=True)
class QuantifiedBrandedDrug[Id: ConceptIdentifier, C: _Concentration](
    DrugNode[Id],
):
    identifier: Id
    unbranded: QuantifiedClinicalDrug[Id, C]
    brand_name: a.BrandName[Id]
    # Redundant, hierarchy builder should check for ancestor consistency
    # unquantified: BrandedDrug[Id, C]
    # contents: SortedTuple[BoundStrength[Id, st.LiquidQuantity]]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.unbranded.get_dose_form()

    @override
    def get_precise_ingredients(self) -> tuple[a.PreciseIngredient | None]:
        return self.unbranded.get_precise_ingredients()

    @override
    def get_brand_name(self) -> a.BrandName[Id]:
        return self.brand_name

    @override
    def get_strength_data(
        self,
    ) -> SortedTuple[BoundStrength[Id, st.LiquidQuantity]]:
        return self.unbranded.get_strength_data()

    @override
    def is_superclass_of(
        self, other: DrugNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        if not passed_hierarchy_checks:
            if not self.unbranded.is_superclass_of(
                other, passed_hierarchy_checks=False
            ):
                return False

        return self.brand_name == other.get_brand_name()
