"""
Contains the individual RxNorm and RxNorm Extension drug classes.
"""

import math  # For NaN checks
from typing import override
from ..rx_model import exception  # For custom exceptions
from ..utils.utils import keep_multiple_values
from ..utils.utils import invert_merge_dict
from ..utils.classes import SortedTuple  # To ensure consistent layout

from dataclasses import dataclass

# Helper classes


class _MulticomponentMixin[Id: "ConceptIdentifier", S: "UnquantifiedStrength"]:
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
            ClinicalDrugComponent[Id, S], Ingredient[Id] | PreciseIngredient
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


# Identifiers
class ConceptId(int):
    """
    Unique identifier for a concept in the OMOP vocabulary.

    This is just a subclass of int, meaning CPython will treat it as an int.
    """


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ConceptCodeVocab:
    """
    Vocabulary and code pair for a concept in the OMOP vocabulary.
    """

    vocabulary_id: str
    concept_code: str

    @override
    def __str__(self):
        return f"{self.vocabulary_id}:{self.concept_code}"


type ConceptIdentifier = ConceptId | ConceptCodeVocab


# Atomic named concepts
@dataclass(frozen=True, order=True, eq=True, slots=True)
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


# # RxNorm
class Ingredient[Id: ConceptIdentifier](__RxAtom[Id]):
    """
    RxNorm or RxNorm Extension ingredient concept.
    """


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


# # Strength/dosage information
@dataclass(frozen=True, order=True, eq=True, slots=True)
class SolidStrength:
    """
    Single value/unit combination for dosage information.
    """

    amount_value: float
    amount_unit: Unit

    def __post_init__(self):
        if self.amount_value < 0:
            raise exception.RxConceptCreationError(
                f"Solid strength must have a non-negative value, not {
                    self.amount_value
                }."
            )

        if math.isnan(self.amount_value):
            raise exception.RxConceptCreationError(
                "Solid strength must have a numeric value, not NaN."
            )

    def unit_matches(self, other: "SolidStrength") -> None:
        if self.amount_unit != other.amount_unit:
            raise exception.StrengthUnitMismatchError(
                f"Solid strength units do not match: "
                f"{self.amount_unit.identifier} "
                f"{self.amount_unit.concept_name} and "
                f"{other.amount_unit.identifier} "
                f"{other.amount_unit.concept_name}.",
                self.amount_unit,
                other.amount_unit,
            )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class LiquidConcentration:
    """
    Dosage given as unquantified concentration.
    """

    numerator_value: float
    numerator_unit: Unit
    # Null for denominator value
    denominator_unit: Unit

    def __post_init__(self):
        if self.numerator_value <= 0:
            raise exception.RxConceptCreationError(
                f"Liquid concentration must have a positive numerator "
                f"value, not {self.numerator_value}."
            )

        if math.isnan(self.numerator_value):
            raise exception.RxConceptCreationError(
                f"Liquid concentration must have a numeric numerator value, "
                f"not {self.numerator_value}."
            )

    # TODO: Concisely check for both numerator and denominator units
    def unit_matches(
        self, other: "LiquidConcentration | LiquidQuantity"
    ) -> None:
        if self.numerator_unit != other.numerator_unit:
            raise exception.StrengthUnitMismatchError(
                f"Liquid quantity numerator units do not match: "
                f"{self.numerator_unit.identifier} "
                f"{self.numerator_unit.concept_name} and "
                f"{other.numerator_unit.identifier} "
                f"{other.numerator_unit.concept_name}.",
                self.numerator_unit,
                other.numerator_unit,
            )

        if self.denominator_unit != other.denominator_unit:
            raise exception.StrengthUnitMismatchError(
                f"Liquid quantity denominator units do not match: "
                f"{self.denominator_unit.identifier} "
                f"{self.denominator_unit.concept_name} and "
                f"{other.denominator_unit.identifier} "
                f"{other.denominator_unit.concept_name}.",
                self.denominator_unit,
                other.denominator_unit,
            )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class LiquidQuantity(LiquidConcentration):
    """
    Quantified liquid dosage with both total content and volume.
    """

    denominator_value: float

    def get_unquantified(self) -> LiquidConcentration:
        return LiquidConcentration(
            numerator_value=self.numerator_value / self.denominator_value,
            numerator_unit=self.numerator_unit,
            denominator_unit=self.denominator_unit,
        )

    def __post_init__(self):
        # Inherit checks from LiquidConcentration
        super().__post_init__()

        if self.denominator_value <= 0:
            raise exception.RxConceptCreationError(
                f"Liquid quantity must have a positive denominator value, not "
                f"{self.denominator_value}."
            )

        if math.isnan(self.denominator_value):
            raise exception.RxConceptCreationError(
                f"Liquid quantity must have a numeric denominator value, not "
                f"{self.denominator_value}."
            )

        if (
            self.numerator_unit == self.denominator_unit
            and self.numerator_value >= self.denominator_value
        ):
            raise exception.RxConceptCreationError(
                f"Liquid quantity must have a numerator value less than the "
                f"denominator value, not {self.numerator_value} >= "
                f"{self.denominator_value}."
            )


type UnquantifiedStrength = SolidStrength | LiquidConcentration


# Derived concepts
# # RxNorm
@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrugComponent[Id: ConceptIdentifier, S: UnquantifiedStrength]:
    """
    Single component containing (precise) ingredient and unquantified strength.
    """

    identifier: Id
    ingredient: Ingredient[Id]
    precise_ingredient: PreciseIngredient | None
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

    def similar_phase(
        self,
        cdc: "ClinicalDrugComponent[ConceptIdentifier, UnquantifiedStrength]",
    ) -> bool:
        """Check if two components have the same strength type."""
        return isinstance(self.strength, type(cdc.strength))


@dataclass(frozen=True, eq=True, slots=True)
class BrandedDrugComponent[Id: ConceptIdentifier, S: UnquantifiedStrength](
    _MulticomponentMixin[Id, S]
):
    """\
Combination of clinical drug components with a stated brand name.

NB: Contains multiple components in one!\
"""

    identifier: Id
    clinical_drug_components: SortedTuple[ClinicalDrugComponent[Id, S]]
    brand_name: BrandName[Id]

    def __post_init__(self):
        self.check_multiple_components(self.clinical_drug_components)


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrugForm[Id: ConceptIdentifier]:
    identifier: Id
    dose_form: DoseForm[Id]
    ingredients: SortedTuple[Ingredient[Id]]

    def __post_init__(self):
        if len(self.ingredients) == 0:
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} must have at least one ingredient."
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


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BrandedDrugForm[Id: ConceptIdentifier]:
    identifier: Id
    clinical_drug_form: ClinicalDrugForm[Id]
    brand_name: BrandName[Id]


# Prescriptable drug classes
# # Clinical drugs have explicit dosage information, that may differ from the
# # components in about 5% window
@dataclass(frozen=True, order=True, eq=True, slots=True)
class BoundStrength[Id: ConceptIdentifier, S: UnquantifiedStrength]:
    ingredient: Ingredient[Id]
    strength: S
    corresponding_component: ClinicalDrugComponent[Id, S]

    def __post_init__(self):
        comp = self.corresponding_component
        if self.ingredient != comp.ingredient:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__} with "
                f"stated ingredient {self.ingredient.identifier} "
                f"{self.ingredient.concept_name}: referenced component "
                f"{comp.identifier} uses ingredient "
                f"{comp.ingredient.identifier} "
                f"{comp.ingredient.concept_name}."
            )

        # Checking for types explicitly, as LiquidConcentration and
        # LiquidQuantity are not subclasses, but may not be mixed
        if type(self.strength) != type(comp.strength):  # noqa: E721
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__}: stated strength "
                f"is of type {type(self.strength).__name__}, but referenced "
                f"component {comp.identifier} uses "
                f"{comp.strength.__class__.__name__}."
            )

        try:
            self.strength.unit_matches(comp.strength)  # pyright: ignore[reportArgumentType]  # noqa: E501
        except exception.StrengthUnitMismatchError as e:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__}: unit mismatch "
                f"between stated strength and referenced component "
                f"{comp.identifier}. {e.args[0]}"
            )

    def similar_phase(
        self,
        cdc: "BoundStrength[ConceptIdentifier, UnquantifiedStrength]",
    ) -> bool:
        """Check if two components have the same strength type."""
        return isinstance(self.strength, type(cdc.strength))


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BoundQuantity[Id: ConceptIdentifier]:
    ingredient: Ingredient[Id]
    strength: LiquidQuantity
    corresponding_component: ClinicalDrugComponent[Id, LiquidConcentration]

    def __post_init__(self):
        comp = self.corresponding_component
        if self.ingredient != comp.ingredient:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__} with "
                f"stated ingredient {self.ingredient.identifier} "
                f"{self.ingredient.concept_name}: referenced component "
                f"{comp.identifier} uses ingredient "
                f"{comp.ingredient.identifier} "
                f"{comp.ingredient.concept_name}."
            )

        # Checking for types explicitly, as LiquidConcentration and
        # LiquidQuantity are not subclasses, but may not be mixed
        if type(self.strength) != type(comp.strength):  # noqa: E721
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__}: stated strength "
                f"is of type {type(self.strength).__name__}, but referenced "
                f"component {comp.identifier} uses "
                f"{comp.strength.__class__.__name__}."
            )

        try:
            self.strength.unit_matches(comp.strength)
        except exception.StrengthUnitMismatchError as e:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__}: unit mismatch "
                f"between stated strength and referenced component "
                f"{comp.identifier}. {e.args[0]}"
            )


# # Unquantified drugs
@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrug[Id: ConceptIdentifier, S: UnquantifiedStrength](
    _MulticomponentMixin[Id, S]
):
    identifier: Id
    form: ClinicalDrugForm[Id]
    contents: SortedTuple[BoundStrength[Id, S]]

    def __post_init__(self):
        self.check_multiple_components(
            SortedTuple(comp.corresponding_component for comp in self.contents)
        )

        first_comp, other_comps = self.contents[0], self.contents[1:]
        if not all(first_comp.similar_phase(comp) for comp in other_comps):
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} {self.identifier} must "
                f"have components of the same phase (solid/liquid)."
            )


@dataclass(frozen=True, order=True, eq=True, slots=True)
class BrandedDrug[Id: ConceptIdentifier, S: UnquantifiedStrength]:
    identifier: Id
    clinical_drug: ClinicalDrug[Id, S]
    brand_name: BrandName[Id]
    # Redundant fields
    branded_form: BrandedDrugForm[Id]
    # TODO: Check if strength is consistent
    content: BrandedDrugComponent[Id, S]

    def __post_init__(self):
        if self.clinical_drug.form != self.branded_form.clinical_drug_form:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__} with "
                f"{self.identifier}: Parent clinical drug form contributes "
                f"form {self.clinical_drug.form.identifier}, but parent "
                f"branded drug contributes form "
                f"{self.branded_form.clinical_drug_form.identifier}."
            )

        # TODO: strength consistency check between explicit contents and
        # branded form. Need to check Vocabularies to determine source of truth


# TODO: implement checks
# # Quantified liquid forms
@dataclass(frozen=True, order=True, eq=True, slots=True)
class QuantifiedClinicalDrug[Id: ConceptIdentifier]:
    identifier: Id
    contents: SortedTuple[BoundQuantity[Id]]
    unquantified_equivalent: ClinicalDrug[Id, LiquidConcentration]


@dataclass(frozen=True, order=True, eq=True, slots=True)
class QuantifiedBrandedDrug[Id: ConceptIdentifier]:
    identifier: Id
    clinical_drug: QuantifiedClinicalDrug[Id]
    unquantified_equivalent: BrandedDrug[Id, LiquidConcentration]
