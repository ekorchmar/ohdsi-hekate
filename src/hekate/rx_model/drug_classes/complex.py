"""
Implementations of complex drug classes
"""

from __future__ import annotations

from abc import ABC, abstractmethod  # For mixins
from dataclasses import dataclass
from typing import override  # For type annotations

from rx_model.drug_classes.base import ConceptIdentifier
import rx_model.drug_classes.strength as st
from rx_model.drug_classes import atom as a
from rx_model.drug_classes.generic import DrugNode  # Generic parent class
from utils.classes import (
    PyRealNumber,  # For strength comparison
    SortedTuple,  # To ensure consistent layout
)
from utils.constants import BOX_SIZE_LIMIT  # For integrity checks of boxes
from utils.enums import ConceptClassId
from utils.exceptions import RxConceptCreationError
from utils.utils import invert_merge_dict, keep_multiple_values


class __MulticomponentMixin[
    Id: ConceptIdentifier,
    S: st.UnquantifiedStrength,
]:
    """Mixin for classes to implement checks with multiple components."""

    identifier: Id

    def check_multiple_components(
        self,
        container: SortedTuple["ClinicalDrugComponent[Id, S]"],
        pi_are_i: bool = False,
    ) -> None:
        """
        Test consistency of multiple components against each other. Intended
        to be called in __post_init__ of classes that contain multiple
        components, whether implicit or explicit.

        Args:
            container: Container of components to test.
            pi_are_i: bool
                If True, treat all precise ingredients as ingredients. Currently
                should not be used.
        """
        if len(container) == 0:
            raise RxConceptCreationError(
                f"{self.__class__.__name__} {self.identifier} must "
                f"have at least one component."
            )

        # Container of length 1 is always valid
        if len(container) == 1:
            return

        # Check for strength type mismatch
        if len(set(cdc.strength.__class__ for cdc in container)) > 1:
            msg = (
                f"{self.__class__.__name__} {self.identifier} contains "
                f"components with different strength types."
            )
            raise RxConceptCreationError(msg)

        # For LiquidConcentration, check that the denominators are the same
        if isinstance(container[0].strength, st.LiquidConcentration):
            denominators: set[a.Unit] = set()

            for cdc in container:
                assert isinstance(cdc.strength, st.LiquidConcentration)
                denominators.add(cdc.strength.denominator_unit)

            if len(denominators) > 1:
                msg = (
                    f"{self.__class__.__name__} {self.identifier} contains "
                    f"components with different denominator units."
                )
                raise RxConceptCreationError(msg)

        duplicate_ingredients: dict[
            ClinicalDrugComponent[Id, S],
            a.Ingredient[Id] | a.PreciseIngredient,
        ]

        if pi_are_i:
            # Check for duplicate ingredients, allowing for precise ingredients
            duplicate_ingredients = keep_multiple_values({
                cdc: cdc.precise_ingredient or cdc.ingredient
                for cdc in container
            })
        else:
            duplicate_ingredients = keep_multiple_values({
                cdc: cdc.ingredient for cdc in container
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
            raise RxConceptCreationError(msg)


# Derived concepts
# # RxNorm
@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrugComponent[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    DrugNode[Id, S]
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
                raise RxConceptCreationError(
                    f"Error creating {self.__class__.__name__} with "
                    f"identifier {self.identifier}: stated precise ingredient "
                    f"{pi.identifier} {pi.concept_name} is not a variant of "
                    f"ingredient {i.identifier} {i.concept_name}."
                )

    @override
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        return [self.precise_ingredient]

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
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
    def get_strength_data(self) -> SortedTuple[st.BoundStrength[Id, S]]:
        return SortedTuple([(self.ingredient, self.strength)])

    @override
    @classmethod
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
    ) -> ClinicalDrugComponent[Id, S]:
        if len(strength_data) != 1:
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have exactly one strength "
                f"entry, not {len(strength_data)}."
            )

        if len(precise_ingredients) > 1:
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have at most one precise "
                f"ingredient, not {len(precise_ingredients)}."
            )

        pi = precise_ingredients.pop() if precise_ingredients else None

        return cls(
            identifier=identifier,
            ingredient=strength_data[0][0],
            strength=strength_data[0][1],
            precise_ingredient=pi,
        )


@dataclass(frozen=True, eq=True, slots=True)
class BrandedDrugComponent[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    __MulticomponentMixin[Id, S], DrugNode[Id, S]
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
    def get_strength_data(self) -> SortedTuple[st.BoundStrength[Id, S]]:
        # Return contents of unwrapped tuples
        return SortedTuple([
            cdc.get_strength_data()[0] for cdc in self.clinical_drug_components
        ])

    @override
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        return [cdc.precise_ingredient for cdc in self.clinical_drug_components]

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            if not all(
                cdc.is_superclass_of_drug_node(
                    other, passed_hierarchy_checks=False
                )
                for cdc in self.clinical_drug_components
            ):
                return False
        else:
            # We are the first jump to multicomponent classes
            if len(self.clinical_drug_components) != len(
                other.get_strength_data()
            ):
                return False

        # Only check the brand name
        return self.brand_name == other.get_brand_name()

    @override
    def get_brand_name(self) -> a.BrandName[Id]:
        return self.brand_name

    @override
    @classmethod
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
    ) -> BrandedDrugComponent[Id, S]:
        if len(strength_data) != len(parents[ConceptClassId.CDC]):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have the same number of "
                f"components as its clinical drug counterpart, but has "
                f"{len(strength_data)} components."
            )

        cdcs: list[ClinicalDrugComponent[Id, S]] = []
        for cdc_node in parents[ConceptClassId.CDC]:
            if not isinstance(cdc_node, ClinicalDrugComponent):
                raise RxConceptCreationError(
                    f"{cls.__name__} {identifier} must have components of "
                    f"type ClinicalDrugComponent, but has {cdc_node}."
                )
            cdc: ClinicalDrugComponent[Id, S] = cdc_node
            cdcs.append(cdc)

        brand_name = attributes[ConceptClassId.BRAND_NAME]
        assert isinstance(brand_name, a.BrandName)
        return cls(
            identifier=identifier,
            clinical_drug_components=SortedTuple(cdcs),
            brand_name=brand_name,
        )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrugForm[Id: ConceptIdentifier](DrugNode[Id, None]):
    identifier: Id
    dose_form: a.DoseForm[Id]
    ingredients: SortedTuple[a.Ingredient[Id]]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.dose_form

    def __post_init__(self):
        if len(self.ingredients) == 0:
            raise RxConceptCreationError(
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
            raise RxConceptCreationError(msg)

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            # Check all ingredients
            if any(
                ing not in self.ingredients
                for ing, _ in other.get_strength_data()
            ):
                return False
        else:
            # We are the first jump to multicomponent classes
            if len(self.ingredients) != len(other.get_strength_data()):
                return False

        # Only check the dose form
        return self.dose_form == other.get_dose_form()

    @override
    def get_strength_data(self) -> SortedTuple[st.BoundStrength[Id, None]]:
        return SortedTuple([(ing, None) for ing in self.ingredients])

    @override
    def get_precise_ingredients(self) -> list[None]:  # pyright: ignore[reportIncompatibleMethodOverride]  # noqa: E501
        return [None] * len(self.ingredients)

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[DrugNode[Id, st.Strength | None]]],
        attributes: dict[
            ConceptClassId, a.BrandName[Id] | a.DoseForm[Id] | a.Supplier[Id]
        ],
        precise_ingredients: list[a.PreciseIngredient],
        strength_data: SortedTuple[st.BoundStrength[Id, None]],
        box_size: int | None,
    ) -> ClinicalDrugForm[Id]:
        dose_form = attributes[ConceptClassId.DOSE_FORM]
        assert isinstance(dose_form, a.DoseForm)
        return cls(
            identifier=identifier,
            dose_form=dose_form,
            ingredients=SortedTuple(ing for ing, _ in strength_data),
        )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BrandedDrugForm[Id: ConceptIdentifier](DrugNode[Id, None]):
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
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            if not self.clinical_drug_form.is_superclass_of_drug_node(
                other, passed_hierarchy_checks=False
            ):
                return False

        # Only check the brand name
        return self.brand_name == other.get_brand_name()

    @override
    def get_strength_data(self) -> SortedTuple[st.BoundStrength[Id, None]]:
        return self.clinical_drug_form.get_strength_data()

    @override
    def get_precise_ingredients(self) -> list[None]:  # pyright: ignore[reportIncompatibleMethodOverride]  # noqa: E501
        return self.clinical_drug_form.get_precise_ingredients()

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[DrugNode[Id, st.Strength | None]]],
        attributes: dict[
            ConceptClassId, a.BrandName[Id] | a.DoseForm[Id] | a.Supplier[Id]
        ],
        precise_ingredients: list[a.PreciseIngredient],
        strength_data: SortedTuple[st.BoundStrength[Id, None]],
        box_size: int | None,
    ) -> BrandedDrugForm[Id]:
        (cdf,) = parents[ConceptClassId.CDF]
        brand_name = attributes[ConceptClassId.BRAND_NAME]
        assert isinstance(brand_name, a.BrandName)
        assert isinstance(cdf, ClinicalDrugForm)
        return cls(
            identifier=identifier,
            clinical_drug_form=cdf,
            brand_name=brand_name,
        )


# Prescriptable drug classes
# # Unquantified drugs
@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrug[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    __MulticomponentMixin[Id, S], DrugNode[Id, S]
):
    identifier: Id
    form: ClinicalDrugForm[Id]
    clinical_drug_components: SortedTuple[ClinicalDrugComponent[Id, S]]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.form.dose_form

    @override
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        return [cdc.precise_ingredient for cdc in self.clinical_drug_components]

    @override
    def get_strength_data(self) -> SortedTuple[st.BoundStrength[Id, S]]:
        return SortedTuple(
            cdc.get_strength_data()[0] for cdc in self.clinical_drug_components
        )

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        # Tests are only ever needed if passed_hierarchy_checks is False,
        # as CD is fully derivative from its components and form
        if not passed_hierarchy_checks:
            if len(self.clinical_drug_components) != len(
                other.get_strength_data()
            ):
                return False

            if not all(
                comp.is_superclass_of_drug_node(other)
                for comp in self.clinical_drug_components
            ):
                return False

            if not self.form.is_superclass_of_drug_node(
                other, passed_hierarchy_checks=False
            ):
                return False
        return True

    @override
    @classmethod
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
    ) -> ClinicalDrug[Id, S]:
        (cdf,) = parents[ConceptClassId.CDF]
        if not isinstance(cdf, ClinicalDrugForm):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"ClinicalDrugForm, but has {cdf}."
            )
        cdc_nodes = parents[ConceptClassId.CDC]
        if len(cdc_nodes) != len(strength_data):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have the same number of "
                f"components as strength data, but has {len(cdc_nodes)} "
                f"components and {len(strength_data)} strength entries."
            )

        cdcs: list[ClinicalDrugComponent[Id, S]] = []
        cdc: ClinicalDrugComponent[Id, S]
        for cdc_node in cdc_nodes:
            assert isinstance(cdc_node, ClinicalDrugComponent)
            cdc = cdc_node
            cdcs.append(cdc)
        return cls(
            identifier=identifier,
            form=cdf,
            clinical_drug_components=SortedTuple(cdcs),
        )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BrandedDrug[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    DrugNode[Id, S]
):
    identifier: Id
    # NOTE: BDs are redundant in their definition, as any 2 of 3 get all the
    # data.
    clinical_drug: ClinicalDrug[Id, S]
    branded_form: BrandedDrugForm[Id]
    branded_component: BrandedDrugComponent[Id, S]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.clinical_drug.form.dose_form

    @override
    def get_brand_name(self) -> a.BrandName[Id]:
        # NOTE: can be taken from either branded ancestor

        # return self.branded_form.brand_name
        return self.branded_component.brand_name

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            # Takes care of strength and dose form checks
            if not self.clinical_drug.is_superclass_of_drug_node(
                other, passed_hierarchy_checks=False
            ):
                return False

        return self.get_brand_name() == other.get_brand_name()

    @override
    def get_strength_data(self) -> SortedTuple[st.BoundStrength[Id, S]]:
        return self.clinical_drug.get_strength_data()

    @override
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        return self.clinical_drug.get_precise_ingredients()

    @override
    @classmethod
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
    ) -> BrandedDrug[Id, S]:
        (cd_node,) = parents[ConceptClassId.CD]
        (bdf_node,) = parents[ConceptClassId.BDF]
        (bdc_node,) = parents[ConceptClassId.BDC]

        cd: ClinicalDrug[Id, S]
        bdf: BrandedDrugForm[Id]
        bdc: BrandedDrugComponent[Id, S]

        if not isinstance(cd_node, ClinicalDrug):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"ClinicalDrug, but has {cd_node}."
            )
        cd = cd_node

        if not isinstance(bdf_node, BrandedDrugForm):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"BrandedDrugForm, but has {bdf_node}."
            )
        bdf = bdf_node

        if not isinstance(bdc_node, BrandedDrugComponent):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"BrandedDrugComponent, but has {bdc_node}."
            )
        bdc = bdc_node

        return cls(
            identifier=identifier,
            clinical_drug=cd,
            branded_form=bdf,
            branded_component=bdc,
        )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class QuantifiedClinicalDrug[Id: ConceptIdentifier, C: st.Concentration](
    DrugNode[Id, st.LiquidQuantity]
):
    identifier: Id
    contents: SortedTuple[st.BoundStrength[Id, st.LiquidQuantity]]
    unquantified: ClinicalDrug[Id, C]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.unquantified.form.dose_form

    @override
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        return self.unquantified.get_precise_ingredients()

    def __post_init__(self):
        # Check strength data consistency
        unique_denoms: set[tuple[PyRealNumber, a.Unit]] = set()
        for _, quantity in self.contents:
            unit: a.Unit = quantity.denominator_unit
            value: PyRealNumber = quantity.denominator_value
            unique_denoms.add((value, unit))

        if len(unique_denoms) > 1:
            raise RxConceptCreationError(
                f"Quantified clinical drug {self.identifier} must have "
                f"consistent denominator units and values."
            )

        # Check consistency against the unquantified form
        if len(self.contents) != len(
            self.unquantified.clinical_drug_components
        ):
            raise RxConceptCreationError(
                f"Quantified clinical drug {self.identifier} must have the "
                f"same number of components as its unquantified counterpart"
                f" {self.unquantified.identifier}."
            )

        shared_iter = zip(
            self.contents, self.unquantified.clinical_drug_components
        )
        for (ing, quantity), cdc in shared_iter:
            u_ing = cdc.ingredient
            strength = cdc.strength

            if ing != u_ing:
                raise RxConceptCreationError(
                    f"Quantified clinical drug {self.identifier} must have "
                    f"same ingredients as its unquantified counterpart."
                )

            if not quantity.matches(strength):
                raise RxConceptCreationError(
                    f"Quantified clinical drug {self.identifier} must have "
                    f"same strength data as its unquantified counterpart for "
                    f"ingredient {ing.identifier} {ing.concept_name}."
                )

    @override
    def get_strength_data(
        self,
    ) -> SortedTuple[st.BoundStrength[Id, st.LiquidQuantity]]:
        return self.contents

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            if not self.unquantified.is_superclass_of_drug_node(
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

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[DrugNode[Id, st.Strength | None]]],
        attributes: dict[
            ConceptClassId, a.BrandName[Id] | a.DoseForm[Id] | a.Supplier[Id]
        ],
        precise_ingredients: list[a.PreciseIngredient],
        strength_data: SortedTuple[st.BoundStrength[Id, st.LiquidQuantity]],
        box_size: int | None,
    ) -> QuantifiedClinicalDrug[Id, C]:
        (cd_node,) = parents[ConceptClassId.CD]
        if not isinstance(cd_node, ClinicalDrug):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"ClinicalDrug, but has {cd_node}."
            )
        cd: ClinicalDrug[Id, C] = cd_node
        return cls(
            identifier=identifier,
            contents=strength_data,
            unquantified=cd,
        )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class QuantifiedBrandedDrug[Id: ConceptIdentifier](
    DrugNode[Id, st.LiquidQuantity],
):
    identifier: Id
    unbranded: QuantifiedClinicalDrug[Id, st.Concentration]
    brand_name: a.BrandName[Id]
    # Redundant, hierarchy builder should check for ancestor consistency
    # unquantified: BrandedDrug[Id, C]
    # contents: SortedTuple[st.BoundStrength[Id, st.LiquidQuantity]]

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        return self.unbranded.get_dose_form()

    @override
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        return self.unbranded.get_precise_ingredients()

    @override
    def get_brand_name(self) -> a.BrandName[Id]:
        return self.brand_name

    @override
    def get_strength_data(
        self,
    ) -> SortedTuple[st.BoundStrength[Id, st.LiquidQuantity]]:
        return self.unbranded.get_strength_data()

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            if not self.unbranded.is_superclass_of_drug_node(
                other, passed_hierarchy_checks=False
            ):
                return False

        return self.brand_name == other.get_brand_name()

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[DrugNode[Id, st.Strength | None]]],
        attributes: dict[
            ConceptClassId, a.BrandName[Id] | a.DoseForm[Id] | a.Supplier[Id]
        ],
        precise_ingredients: list[a.PreciseIngredient],
        strength_data: SortedTuple[st.BoundStrength[Id, st.LiquidQuantity]],
        box_size: int | None,
    ) -> QuantifiedBrandedDrug[Id]:
        qcd_node = parents[ConceptClassId.QCD][0]
        if not isinstance(qcd_node, QuantifiedClinicalDrug):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"QuantifiedClinicalDrug, but has {qcd_node}."
            )
        qcd: QuantifiedClinicalDrug[Id, st.Concentration] = qcd_node
        brand_name = attributes[ConceptClassId.BRAND_NAME]
        assert isinstance(brand_name, a.BrandName)
        return cls(
            identifier=identifier,
            unbranded=qcd,
            brand_name=brand_name,
        )


class __Boxed[
    Id: ConceptIdentifier,
    S: st.Strength,
](DrugNode[Id, S], ABC):
    """
    Metaclass for shared behavior of clinical box-size defining classes.
    """

    identifier: Id
    unboxed: DrugNode[Id, S]

    @override
    @abstractmethod
    def get_box_size(self) -> int: ...

    @override
    def get_strength_data(
        self,
    ) -> SortedTuple[st.BoundStrength[Id, S]]:
        return self.unboxed.get_strength_data()

    @override
    def get_precise_ingredients(self) -> list[a.PreciseIngredient | None]:
        return self.unboxed.get_precise_ingredients()

    @override
    def get_dose_form(self) -> a.DoseForm[Id]:
        df = self.unboxed.get_dose_form()
        assert df is not None
        return df

    @override
    def get_brand_name(self) -> a.BrandName[Id] | None:
        return self.unboxed.get_brand_name()


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrugBox[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    __Boxed[Id, S]
):
    identifier: Id
    unboxed: ClinicalDrug[Id, S]  # pyright: ignore[reportIncompatibleVariableOverride]  # noqa: E501
    box_size: int

    @override
    def get_box_size(self) -> int:
        return self.box_size

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            if not self.unboxed.is_superclass_of_drug_node(
                other, passed_hierarchy_checks=False
            ):
                return False

        # Box size can only match exactly
        return self.get_box_size() == other.get_box_size()

    def __post_init__(self):
        if BOX_SIZE_LIMIT < self.get_box_size() <= 0:
            raise RxConceptCreationError(
                f"Box size of {self.__class__.__name__} {self.identifier} must "
                f"be a positive integer less than or equal to "
                f"{BOX_SIZE_LIMIT}, not {self.get_box_size()}."
            )

    @override
    @classmethod
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
    ) -> ClinicalDrugBox[Id, S]:
        if box_size is None:
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a box size defined."
            )
        (cd_node,) = parents[ConceptClassId.CD]
        if not isinstance(cd_node, ClinicalDrug):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"ClinicalDrug, but has {cd_node}."
            )
        cd: ClinicalDrug[Id, S] = cd_node
        return cls(
            identifier=identifier,
            unboxed=cd,
            box_size=box_size,
        )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class QuantifiedClinicalBox[Id: ConceptIdentifier, C: st.Concentration](
    __Boxed[Id, st.LiquidQuantity],
):
    identifier: Id
    unboxed: QuantifiedClinicalDrug[Id, C]  # pyright: ignore[reportIncompatibleVariableOverride]  # noqa: E501
    unquantified: ClinicalDrugBox[Id, C]
    # NOTE: integrity checks between unboxed and unquantified contributing nodes
    # have to be handled by hierarchy builder

    @override
    def get_box_size(self) -> int:
        return self.unquantified.box_size

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            for node in (self.unboxed, self.unquantified):
                if not node.is_superclass_of_drug_node(
                    other, passed_hierarchy_checks=False
                ):
                    return False

        # Box size can only match exactly
        return self.get_box_size() == other.get_box_size()

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[DrugNode[Id, st.Strength | None]]],
        attributes: dict[
            ConceptClassId, a.BrandName[Id] | a.DoseForm[Id] | a.Supplier[Id]
        ],
        precise_ingredients: list[a.PreciseIngredient],
        strength_data: SortedTuple[st.BoundStrength[Id, st.LiquidQuantity]],
        box_size: int | None,
    ) -> QuantifiedClinicalBox[Id, C]:
        (qcd_node,) = parents[ConceptClassId.QCD]
        if not isinstance(qcd_node, QuantifiedClinicalDrug):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"QuantifiedClinicalDrug, but has {qcd_node}."
            )
        qcd: QuantifiedClinicalDrug[Id, C] = qcd_node

        (cdb_node,) = parents[ConceptClassId.CDB]
        if not isinstance(cdb_node, ClinicalDrugBox):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"ClinicalDrugBox, but has {cdb_node}."
            )
        cdb: ClinicalDrugBox[Id, C] = cdb_node

        return cls(
            identifier=identifier,
            unboxed=qcd,
            unquantified=cdb,
        )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BrandedDrugBox[Id: ConceptIdentifier, S: st.UnquantifiedStrength](
    __Boxed[Id, S]
):
    identifier: Id
    unboxed: BrandedDrug[Id, S]  # pyright: ignore[reportIncompatibleVariableOverride]  # noqa: E501
    clinical_box: ClinicalDrugBox[Id, S]

    @override
    def get_box_size(self) -> int:
        return self.clinical_box.box_size

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            for node in (self.unboxed, self.clinical_box):
                if not node.is_superclass_of_drug_node(
                    other, passed_hierarchy_checks=False
                ):
                    return False
        # BDB is redundant to its parents, so hierarchy checks are all we need
        return True

    @override
    @classmethod
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
    ) -> BrandedDrugBox[Id, S]:
        if box_size is None:
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a box size defined."
            )
        (bd_node,) = parents[ConceptClassId.BD]
        (cdb_node,) = parents[ConceptClassId.CDB]
        if not isinstance(bd_node, BrandedDrug):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"Branded Drug, but has {bd_node}."
            )
        if not isinstance(cdb_node, ClinicalDrugBox):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"ClinicalDrugBox, but has {cdb_node}."
            )
        bd: BrandedDrug[Id, S] = bd_node
        cdb: ClinicalDrugBox[Id, S] = cdb_node
        return cls(
            identifier=identifier,
            unboxed=bd,
            clinical_box=cdb,
        )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class QuantifiedBrandedBox[Id: ConceptIdentifier, C: st.Concentration](
    __Boxed[Id, st.LiquidQuantity],
):
    identifier: Id
    unboxed: QuantifiedBrandedDrug[Id]  # pyright: ignore[reportIncompatibleVariableOverride]  # noqa: E501
    quant_clinical_box: QuantifiedClinicalBox[Id, C]

    @override
    def get_box_size(self) -> int:
        return self.quant_clinical_box.get_box_size()

    @override
    def is_superclass_of_drug_node(
        self,
        other: DrugNode[Id, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> bool:
        if not passed_hierarchy_checks:
            for node in (self.unboxed, self.quant_clinical_box):
                if not node.is_superclass_of_drug_node(
                    other, passed_hierarchy_checks=False
                ):
                    return False
        # BDB is redundant to its parents, so hierarchy checks are all we need
        return True

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: Id,
        parents: dict[ConceptClassId, list[DrugNode[Id, st.Strength | None]]],
        attributes: dict[
            ConceptClassId, a.BrandName[Id] | a.DoseForm[Id] | a.Supplier[Id]
        ],
        precise_ingredients: list[a.PreciseIngredient],
        strength_data: SortedTuple[st.BoundStrength[Id, st.LiquidQuantity]],
        box_size: int | None,
    ) -> QuantifiedBrandedBox[Id, C]:
        if box_size is None:
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a box size defined."
            )
        (qcb_node,) = parents[ConceptClassId.QCB]
        (qbd_node,) = parents[ConceptClassId.QBD]
        if not isinstance(qcb_node, QuantifiedClinicalBox):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"Branded Drug, but has {qcb_node}."
            )
        if not isinstance(qbd_node, QuantifiedBrandedDrug):
            raise RxConceptCreationError(
                f"{cls.__name__} {identifier} must have a component of type "
                f"ClinicalDrugBox, but has {qbd_node}."
            )
        qbd: QuantifiedBrandedDrug[Id] = qbd_node
        qcb: QuantifiedClinicalBox[Id, C] = qcb_node
        return cls(
            identifier=identifier,
            unboxed=qbd,
            quant_clinical_box=qcb,
        )
