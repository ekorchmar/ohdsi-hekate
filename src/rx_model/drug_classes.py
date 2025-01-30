"""Contains the individual RxNorm and RxNorm Extension drug classes."""
from dataclasses import dataclass  # For dataclass definitions
import math  # For NaN checks
from typing import Optional  # For optional fields in dataclasses
from typing import Self
from rxmodel import exception  # For custom exceptions
from src.utils import utils  # For utility functions in integrity checks


# Helper classes
class _MulticomponentMixin:
    """Mixin for classes to implement checks with multiple components."""

    def check_multiple_components(
            self, container: tuple["ClinicalDrugComponent"]
    ) -> None:
        if len(container) == 0:
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} CONCEPT_ID {self.concept_id} must "
                f"have at least one component."
            )

        # Check for duplicate ingredients
        I, PI, CDC = Ingredient, PreciseIngredient, ClinicalDrugComponent
        duplicate_ingredients: dict[CDC, I | PI] = utils.keep_multiple_values({
            cdc: cdc.precise_ingredient or cdc.ingredient
            for cdc in container
        })

        if duplicate_ingredients:
            msg = (
                f"{self.__class__.__name__} CONCEPT_ID {self.concept_id} "
                f"contains duplicate ingredients:"
            )
            for ing, cdcs in utils.invert_dict(duplicate_ingredients).items():
                msg += f" {ing.concept_id} {ing.concept_name}"
                msg += f" ({ing.__class__.__name__}), coming from: "
                msg += " and ".join(str(cdc.concept_id) for cdc in cdcs)
                msg += ";"
            raise exception.RxConceptCreationError(msg)


# Atomic named concepts
@dataclass(frozen=True, slots=True)
class __RxAtom:
    """A single atomic concept in the RxNorm vocabulary."""
    concept_id: int
    concept_name: str

    def __post_init__(self):
        if not self.concept_name:
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} CONCEPT_ID {
                    self.concept_id}: name must not be empty."
            )


# # RxNorm
Ingredient = __RxAtom
BrandName = __RxAtom
DoseForm = __RxAtom
# # UCUM
Unit = __RxAtom


@dataclass(frozen=True, slots=True)
class PreciseIngredient(__RxAtom):
    invariant: Ingredient


# # RxNorm Extension
Supplier = __RxAtom


# # Strength/dosage information
@dataclass(frozen=True, slots=True)
class SolidStrength:
    """Single value/unit combination for dosage information."""
    amount_value: float
    amount_unit: Unit

    def __post_init__(self):
        if self.amount_value < 0:
            raise exception.RxConceptCreationError(
                f"Solid strength must have a non-negative value, not {
                    self.amount_value}."
            )

        if math.isnan(self.amount_value):
            raise exception.RxConceptCreationError(
                "Solid strength must have a numeric value, not NaN."
            )

    def unit_matches(self, other: "SolidStrength") -> None:
        if self.amount_unit != other.amount_unit:
            raise exception.StrengthUnitMismatchError(
                f"Solid strength units do not match: "
                f"{self.amount_unit.concept_id} "
                f"{self.amount_unit.concept_name} and "
                f"{other.amount_unit.concept_id} "
                f"{other.amount_unit.concept_name}.",
                self.amount_unit,
                other.amount_unit,
            )


@dataclass(frozen=True, slots=True)
class LiquidConcentration:
    """Dosage given as unquantified concentration."""
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
                "Liquid concentration must have a numeric numerator value, "
                "not NaN."
            )

    # TODO: Concisely check for both numerator and denominator units
    def unit_matches(self, other: Self) -> None:
        if self.numerator_unit != other.numerator_unit:
            raise exception.StrengthUnitMismatchError(
                f"Liquid quantity numerator units do not match: "
                f"{self.numerator_unit.concept_id} "
                f"{self.numerator_unit.concept_name} and "
                f"{other.numerator_unit.concept_id} "
                f"{other.numerator_unit.concept_name}.",
                self.numerator_unit,
                other.numerator_unit,
            )

        if self.denominator_unit != other.denominator_unit:
            raise exception.StrengthUnitMismatchError(
                f"Liquid quantity denominator units do not match: "
                f"{self.denominator_unit.concept_id} "
                f"{self.denominator_unit.concept_name} and "
                f"{other.denominator_unit.concept_id} "
                f"{other.denominator_unit.concept_name}.",
                self.denominator_unit,
                other.denominator_unit,
            )


@dataclass(frozen=True, slots=True)
class LiquidQuantity(LiquidConcentration):
    """Quantified liquid dosage with both total content and volume."""
    denominator_value: float

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
                "Liquid quantity must have a numeric denominator value, not "
                "NaN."
            )

        if (
                self.numerator_unit == self.denominator_unit and
                self.numerator_value >= self.denominator_value
        ):
            raise exception.RxConceptCreationError(
                f"Liquid quantity must have a numerator value less than the "
                f"denominator value, not {self.numerator_value} >= "
                f"{self.denominator_value}."
            )


type UnquantifiedStrength = SolidStrength | LiquidConcentration


# Derived concepts
# # RxNorm
@dataclass(frozen=True, slots=True)
class ClinicalDrugComponent:
    """\
Single component containing (precise) ingredient and unquantified strength.\
"""
    concept_id: int
    ingredient: Ingredient
    precise_ingredient: Optional[PreciseIngredient]
    strength: UnquantifiedStrength

    def __post_init__(self):
        if self.precise_ingredient is not None:
            pi, i = self.precise_ingredient, self.ingredient
            if pi.invariant != i:
                raise exception.RxConceptCreationError(
                    f"Error creating {self.__class__.__name__} with "
                    f"CONCEPT_ID {self.concept_id}: stated precise ingredient "
                    f"{pi.concept_id} {pi.concept_name} is not a variant of "
                    f"ingredient {i.concept_id} {i.concept_name}."
                )

    def similar_phase(self, other: Self) -> bool:
        """Check if two components have the same strength type."""
        return isinstance(self.strength, type(other.strength))


@dataclass(frozen=True, slots=True)
class BrandedDrugComponent(_MulticomponentMixin):
    """\
Combination of clinical drug components with a stated brand name.

NB: Contains multiple components in one!\
"""
    concept_id: int
    clinical_drug_components: tuple[ClinicalDrugComponent]
    brand_name: BrandName

    def __post_init__(self):
        self.check_multiple_components(self.clinical_drug_components)


@dataclass(frozen=True, slots=True)
class ClinicalDrugForm:
    concept_id: int
    dose_form: DoseForm
    ingredients: tuple[Ingredient]

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
                f"{self.__class__.__name__} CONCEPT_ID {
                    self.concept_id} contains "
                f"duplicate ingredients:"
            )
            for ing, cnt in counts.items():
                if cnt > 1:
                    msg += f" {ing.concept_id} {ing.concept_name}"
                    msg += f"({cnt});"
            raise exception.RxConceptCreationError(msg)


@dataclass(frozen=True, slots=True)
class BrandedDrugForm:
    concept_id: int
    clinical_drug_form: ClinicalDrugForm
    brand_name: BrandName


# Prescriptable drug classes
# # Clinical drugs have explicit dosage information, that may differ from the
# # components in about 5% window
@dataclass(frozen=True, slots=True)
class BoundUnquantifiedStrength:
    ingredient: Ingredient
    strength: UnquantifiedStrength
    corresponding_component: ClinicalDrugComponent

    def __post_init__(self):
        comp = self.corresponding_component
        if self.ingredient != comp.ingredient:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__} with "
                f"stated ingredient {self.ingredient.concept_id} "
                f"{self.ingredient.concept_name}: referenced component "
                f"{comp.concept_id} uses ingredient "
                f"{comp.ingredient.concept_id} "
                f"{comp.ingredient.concept_name}."
            )

        # Checking for types explicitly, as LiquidConcentration and
        # LiquidQuantity are not subclasses, but may not be mixed
        if type(self.strength) != type(comp.strength):  # noqa: E721
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__}: stated strength "
                f"is of type {type(self.strength).__name__}, but referenced "
                f"component {comp.concept_id} uses "
                f"{comp.strength.__class__.__name__}."
            )

        try:
            self.strength.unit_matches(comp.strength)
        except exception.StrengthUnitMismatchError as e:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__}: unit mismatch "
                f"between stated strength and referenced component "
                f"{comp.concept_id}. {e.args[0]}"
            )


@dataclass(frozen=True, slots=True)
class BoundQuantifiedStrength:
    ingredient: Ingredient
    strength: LiquidQuantity
    corresponding_component: ClinicalDrugComponent

    def __post_init__(self):
        comp = self.corresponding_component
        if self.ingredient != comp.ingredient:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__} with "
                f"stated ingredient {self.ingredient.concept_id} "
                f"{self.ingredient.concept_name}: referenced component "
                f"{comp.concept_id} uses ingredient "
                f"{comp.ingredient.concept_id} "
                f"{comp.ingredient.concept_name}."
            )

        # Checking for types explicitly, as LiquidConcentration and
        # LiquidQuantity are not subclasses, but may not be mixed
        if type(self.strength) != type(comp.strength):  # noqa: E721
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__}: stated strength "
                f"is of type {type(self.strength).__name__}, but referenced "
                f"component {comp.concept_id} uses "
                f"{comp.strength.__class__.__name__}."
            )

        try:
            self.strength.unit_matches(comp.strength)
        except exception.StrengthUnitMismatchError as e:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__}: unit mismatch "
                f"between stated strength and referenced component "
                f"{comp.concept_id}. {e.args[0]}"
            )


# # Unquantified drugs
@dataclass(frozen=True, slots=True)
class ClinicalDrug(_MulticomponentMixin):
    concept_id: int
    form: ClinicalDrugForm
    contents: tuple[BoundUnquantifiedStrength]

    def __post_init__(self):
        self.check_multiple_components(
            tuple(comp.corresponding_component for comp in self.contents)
        )

        first_comp, other_comps = self.contents[0], self.contents[1:]
        if not all(first_comp.similar_phase(comp) for comp in other_comps):
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} CONCEPT_ID {self.concept_id} must "
                f"have components of the same phase (solid/liquid)."
            )


@dataclass(frozen=True, slots=True)
class BrandedDrug:
    concept_id: int
    clinical_drug: ClinicalDrug
    brand_name: BrandName
    # Redundant fields
    branded_form: BrandedDrugForm
    content: BrandedDrugComponent  # TODO: Check if strength is consistent

    def __post_init__(self):
        if self.clinical_drug.form != self.branded_form.clinical_drug_form:
            raise exception.RxConceptCreationError(
                f"Error creating {self.__class__.__name__} with CONCEPT_ID "
                f"{self.concept_id}: Parent clinical drug form contributes "
                f"form {self.clinical_drug.form.concept_id}, but parent "
                f"branded drug contributes form "
                f"{self.branded_form.clinical_drug_form.concept_id}."
            )

        # TODO: strength consistency check between explicit contents and
        # branded form. Need to check Vocabularies to determine source of truth


# # Quantified forms
@dataclass(frozen=True, slots=True)
class QuantifiedClinicalDrug:
    concept_id: int
    contents: tuple[BoundQuantifiedStrength]
    unquantified_equivalent: ClinicalDrug


@dataclass(frozen=True, slots=True)
class QuantifiedBrandedDrug:
    concept_id: int
    clinical_drug: QuantifiedClinicalDrug
    unquantified_equivalent: BrandedDrug
