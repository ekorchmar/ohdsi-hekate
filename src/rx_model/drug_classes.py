"""Contains the individual RxNorm and RxNorm Extension drug classes."""
import math  # For NaN checks
from typing import Optional  # For optional fields in dataclasses
from typing import Self
from rx_model import exception  # For custom exceptions
from utils import utils  # For utility functions in integrity checks
from utils.classes import SortedTuple
from utils.classes import elementary_dataclass
from utils.classes import complex_dataclass


# Helper classes
class _MulticomponentMixin:
    """Mixin for classes to implement checks with multiple components."""

    def check_multiple_components(
            self, container: SortedTuple["ClinicalDrugComponent"]
    ) -> None:
        if len(container) == 0:
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} {self.identifier} must "
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
                f"{self.__class__.__name__} {self.identifier} "
                f"contains duplicate ingredients:"
            )
            for ing, cdcs in utils.invert_dict(duplicate_ingredients).items():
                msg += f" {ing.identifier} {ing.concept_name}"
                msg += f" ({ing.__class__.__name__}), coming from: "
                msg += " and ".join(str(cdc.identifier) for cdc in cdcs)
                msg += ";"
            raise exception.RxConceptCreationError(msg)


# Identifiers
class ConceptId(int):
    """Unique identifier for a concept in the OMOP vocabulary."""


@elementary_dataclass
class ConceptCodeVocab:
    """\
Vocabulary and code pair for a concept in the OMOP vocabulary.\
"""
    vocabulary_id: str
    concept_code: str

    def __str__(self):
        return f"{self.vocabulary_id}:{self.concept_code}"


type ConceptIdentifier = ConceptId | ConceptCodeVocab


# Atomic named concepts
@elementary_dataclass
class __RxAtom:
    """A single atomic concept in the RxNorm vocabulary."""
    identifier: ConceptIdentifier
    concept_name: str

    def __post_init__(self):
        if not self.concept_name:
            raise exception.RxConceptCreationError(
                f"{self.__class__.__name__} "
                f"{self.identifier}: name must not be empty."
            )


# # RxNorm
class Ingredient(__RxAtom):
    """\
    RxNorm or RxNorm Extension ingredient concept.\
"""


class BrandName(__RxAtom):
    """\
    RxNorm or RxNorm Extension brand name concept.\
"""


class DoseForm(__RxAtom):
    """\
    RxNorm or RxNorm Extension dose form concept.\
"""


# # UCUM
class Unit(__RxAtom):
    """\
    UCUM unit concept used in drug dosage information.\
"""


@complex_dataclass
class PreciseIngredient(__RxAtom):
    invariant: Ingredient


# # RxNorm Extension
class Supplier(__RxAtom):
    """\
    RxNorm Extension supplier concept.\
"""


# # Strength/dosage information
@elementary_dataclass
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
                f"{self.amount_unit.identifier} "
                f"{self.amount_unit.concept_name} and "
                f"{other.amount_unit.identifier} "
                f"{other.amount_unit.concept_name}.",
                self.amount_unit,
                other.amount_unit,
            )


@elementary_dataclass
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


@elementary_dataclass
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
@elementary_dataclass
class ClinicalDrugComponent:
    """\
Single component containing (precise) ingredient and unquantified strength.\
"""
    identifier: ConceptIdentifier
    ingredient: Ingredient
    precise_ingredient: Optional[PreciseIngredient]
    strength: UnquantifiedStrength

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

    def similar_phase(self, other: Self) -> bool:
        """Check if two components have the same strength type."""
        return isinstance(self.strength, type(other.strength))


@complex_dataclass
class BrandedDrugComponent(_MulticomponentMixin):
    """\
Combination of clinical drug components with a stated brand name.

NB: Contains multiple components in one!\
"""
    identifier: ConceptIdentifier
    clinical_drug_components: tuple[ClinicalDrugComponent]
    brand_name: BrandName

    def __post_init__(self):
        self.check_multiple_components(self.clinical_drug_components)


@complex_dataclass
class ClinicalDrugForm:
    identifier: ConceptIdentifier
    dose_form: DoseForm
    ingredients: SortedTuple[Ingredient]

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


@complex_dataclass
class BrandedDrugForm:
    identifier: ConceptIdentifier
    clinical_drug_form: ClinicalDrugForm
    brand_name: BrandName


# Prescriptable drug classes
# # Clinical drugs have explicit dosage information, that may differ from the
# # components in about 5% window
@elementary_dataclass
class BoundUnquantifiedStrength:
    ingredient: Ingredient
    strength: UnquantifiedStrength
    corresponding_component: ClinicalDrugComponent

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


@elementary_dataclass
class BoundQuantifiedStrength:
    ingredient: Ingredient
    strength: LiquidQuantity
    corresponding_component: ClinicalDrugComponent

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
@elementary_dataclass
class ClinicalDrug(_MulticomponentMixin):
    identifier: ConceptIdentifier
    form: ClinicalDrugForm
    contents: tuple[BoundUnquantifiedStrength]

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


@elementary_dataclass
class BrandedDrug:
    identifier: ConceptIdentifier
    clinical_drug: ClinicalDrug
    brand_name: BrandName
    # Redundant fields
    branded_form: BrandedDrugForm
    content: BrandedDrugComponent  # TODO: Check if strength is consistent

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


# # Quantified liquid forms
@elementary_dataclass
class QuantifiedClinicalDrug:
    identifier: ConceptIdentifier
    contents: SortedTuple[BoundQuantifiedStrength]
    unquantified_equivalent: ClinicalDrug


@elementary_dataclass
class QuantifiedBrandedDrug:
    identifier: ConceptIdentifier
    clinical_drug: QuantifiedClinicalDrug
    unquantified_equivalent: BrandedDrug
