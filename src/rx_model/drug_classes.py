"""Contains the individual RxNorm and RxNorm Extension drug classes."""
from typing import Optional
from dataclasses import dataclass
from rxmodel import exception

# Atomic named concepts


@dataclass(frozen=True, slots=True)
class __RxAtom:
    """A single atomic concept in the RxNorm vocabulary."""
    concept_id: int
    concept_name: str


# # RxNorm/UCUM
Ingredient = __RxAtom
BrandName = __RxAtom
DoseForm = __RxAtom
Unit = __RxAtom


@dataclass(frozen=True, slots=True)
class PreciseIngredient(__RxAtom):
    invariant: Ingredient


# # RxNorm Extension
Supplier = __RxAtom


# # Strength/dosage information
@dataclass(frozen=True, slots=True)
class SolidStrength:
    amount_value: float
    amount_unit: Unit


@dataclass(frozen=True, slots=True)
class LiquidConcentration:
    numerator_value: float
    numerator_unit: Unit
    # Null for denominator value
    denominator_unit: Unit


@dataclass(frozen=True, slots=True)
class LiquidQuantity(LiquidConcentration):
    denominator_value: float


type UnquantifiedStrength = SolidStrength | LiquidConcentration


# Derived concepts
# # RxNorm
@dataclass(frozen=True, slots=True)
class ClinicalDrugComponent:
    concept_id: int
    ingredient: Ingredient
    precise_ingredient: Optional[PreciseIngredient]
    strength: UnquantifiedStrength

    def __post_init__(self):
        if self.precise_ingredient is not None:
            pi, i = self.precise_ingredient, self.ingredient
            if pi.invariant != i:
                raise exception.RxConceptCreationError(
                    f"{self.__class__.__name__} uses {pi.concept_id} "
                    f"{pi.concept_name} as its precise ingredient, which is "
                    f"not a variant of the ingredient {i.concept_id} "
                    f"{i.concept_name}."
                )


@dataclass(frozen=True, slots=True)
class BrandedDrugComponent:
    """NB: Contains multiple components in one!"""
    concept_id: int
    clinical_drug_components: tuple[ClinicalDrugComponent]
    brand_name: BrandName


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


@dataclass(frozen=True, slots=True)
class BoundQuantifiedStrength:
    ingredient: Ingredient
    strength: LiquidQuantity


# # Unquantified drugs
@dataclass(frozen=True, slots=True)
# TODO: Split in solid and liquid forms?
class ClinicalDrug:
    concept_id: int
    form: ClinicalDrugForm
    contents: tuple[BoundUnquantifiedStrength]


@dataclass(frozen=True, slots=True)
class BrandedDrug:
    concept_id: int
    clinical_drug: ClinicalDrug
    brand_name: BrandName
    # Redundant fields
    form: BrandedDrugForm
    content: BrandedDrugComponent  # TODO: Check if strength is consistent


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
