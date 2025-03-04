"""
Contains the individual RxNorm and RxNorm Extension drug classes.

The classes do not implement any external integrity checks, e.g. verifying
consistency between DRUG_STRENGTH data for a Branded Drug and its Clinical Drug
ancestor.

However, if any checks can operate on class attributes themselves, they are
implemented. For example, the `SolidStrength` class checks that the values are
non-negative and not NaN.
"""

import math  # For NaN checks
from abc import (  # For abstract interfaces
    ABC,
    abstractmethod,
)
from collections.abc import Sequence  # For type annotations
from dataclasses import dataclass
from typing import override  # For type annotations

from rx_model import exception  # For custom exceptions
from utils.classes import SortedTuple  # To ensure consistent layout
from utils.constants import (
    PERCENT_CONCEPT_ID,  # For gaseous percentage check
)

# For strength matching
from utils.constants import STRENGTH_CLOSURE_BOUNDARY_HIGH as _HIGH
from utils.constants import STRENGTH_CLOSURE_BOUNDARY_LOW as _LOW
from utils.utils import invert_merge_dict, keep_multiple_values

# Helper classes
type BoundStrength[Id: "ConceptIdentifier", S: "_StrengthMeta | None"] = tuple[
    "Ingredient[Id]", S
]


class DrugNode[Id: "ConceptIdentifier"](ABC):
    """
    Metaclass for the nodes in the drug concept hierarchy.

    Purpose of this class is to provide a consistent interface for the
    transitive closure methods, allowing dynamic typing to be used.
    """

    identifier: Id

    @abstractmethod
    def is_superclass_of(
        self, other: "DrugNode[Id]", passed_hierarchy_checks: bool = True
    ) -> bool:
        """
        Check if this node is a superclass of another node.

        Args:
            other: The node to check against.
            passed_hierarchy_checks: Whether the tested node had already passed
                the corresponding checks by the predecessors of this node. This
                is used to avoid redundant checks in the hierarchy.
        """

    @abstractmethod
    def get_strength_data(
        self,
    ) -> SortedTuple[BoundStrength[Id, "_StrengthMeta | None"]]:
        """
        Retrieve all strength data for this node. Every entry will always
        specify an ingredient and a strength, which may be None for nodes
        that do not have strength data (e.g. ingredients).
        """

    @abstractmethod
    def get_precise_ingredients(self) -> "Sequence[PreciseIngredient | None]":
        """
        Retrieve all Precise Ingredients participating in this node. Data
        will be returned as a sequence of PreciseIngredient instances or None,
        matching layout with the `get_strength_data` method.
        """

    # Methods to get a possibly inherited attribute
    # NOTE: These methods default to returning None. Be careful in subclasses!
    def get_brand_name(self) -> "BrandName[Id] | None":
        """
        Retrieve the brand name for this node.
        """
        return None

    def get_dose_form(self) -> "DoseForm[Id] | None":
        """
        Retrieve the dose form for this node.
        """
        return None

    def get_supplier(self) -> "Supplier[Id] | None":
        """
        Retrieve the supplier for this node.
        """
        return None


class _StrengthMeta(ABC):
    """
    Abstract class for unquantified strength classes.

    This class is used to provide a consistent interface for the strength
    classes, allowing dynamic typing to be used.
    """

    @abstractmethod
    def get_unquantified(self) -> "UnquantifiedStrength":
        """
        Retrieve the unquantified strength from a quantified strength. May
        return itself for unquantified strengths.
        """

    @abstractmethod
    def _unit_matches(self, other: "_StrengthMeta") -> bool:
        """
        Check if the units of two strength instances match.
        """

    @abstractmethod
    def _values_match(self, other: "_StrengthMeta") -> bool:
        """
        Check if the strength values of two strength instances match.

        NB: This check is separate from the unit checks.
        """

    def matches(self, other: "_StrengthMeta") -> bool:
        """
        Check if two strength instances match.

        This method unifies the unit identity check and the value comparison
        checks.
        """
        return self._unit_matches(other) and self._values_match(other)


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


# # Strength/dosage information
@dataclass(frozen=True, order=True, eq=True, slots=True)
class SolidStrength(_StrengthMeta):
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

    @override
    def get_unquantified(self) -> "SolidStrength":
        return self

    @override
    def _unit_matches(self, other: _StrengthMeta) -> bool:
        if not isinstance(other, SolidStrength):
            return False

        return self.amount_unit == other.amount_unit

    @override
    def _values_match(self, other: _StrengthMeta) -> bool:
        assert isinstance(other, SolidStrength)
        diff = self.amount_value / other.amount_value
        return _LOW <= diff <= _HIGH


@dataclass(frozen=True, order=True, eq=True, slots=True)
class LiquidConcentration(_StrengthMeta):
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

    @override
    def get_unquantified(self) -> "LiquidConcentration":
        return self

    @override
    def _unit_matches(self, other: _StrengthMeta) -> bool:
        if not isinstance(other.get_unquantified(), LiquidConcentration):
            return False

        assert isinstance(other, LiquidConcentration)

        return (self.numerator_unit, self.denominator_unit) == (
            other.numerator_unit,
            other.denominator_unit,
        )

    @override
    def _values_match(self, other: _StrengthMeta) -> bool:
        other = other.get_unquantified()
        assert isinstance(other, LiquidConcentration)
        diff = self.numerator_value / other.numerator_value
        return _LOW <= diff <= _HIGH


@dataclass(frozen=True, order=True, eq=True, slots=True)
class GaseousPercentage(_StrengthMeta):
    """
    Special case of concentration for gases. Only specifies numerator value
    and unit.
    """

    numerator_value: float
    numerator_unit: Unit
    # Null for denominator value
    # Null for denominator unit

    def __post_init__(self):
        # Numerator unit must be a percentage
        if self.numerator_unit.identifier != ConceptId(PERCENT_CONCEPT_ID):
            raise exception.RxConceptCreationError(
                f"Gaseous concentration must be percent ({PERCENT_CONCEPT_ID})"
                f"percentage unit, not {self.numerator_unit}."
            )

        # Numerator value must be a positive number < 100
        if 0 <= self.numerator_value > 100:
            raise exception.RxConceptCreationError(
                f"Gaseous concentration must have a positive numerator "
                f"value not exceeding 100, not {self.numerator_value}."
            )

    @override
    def get_unquantified(self) -> "GaseousPercentage":
        return self

    @override
    def _unit_matches(self, other: _StrengthMeta) -> bool:
        # There is only one valid unit for gaseous percentages
        return isinstance(other.get_unquantified(), GaseousPercentage)

    @override
    def _values_match(self, other: _StrengthMeta) -> bool:
        other = other.get_unquantified()
        assert isinstance(other, GaseousPercentage)
        diff = self.numerator_value / other.numerator_value
        return _LOW <= diff <= _HIGH


@dataclass(frozen=True, order=True, eq=True, slots=True)
class LiquidQuantity(_StrengthMeta):
    """
    Quantified liquid or gaseous dosage with both total content and volume.
    """

    # NOTE: we are an implicit subclass of LiquidConcentration and
    # GaseousPercentage, so we allow accessing their protected methods here.

    numerator_value: float
    numerator_unit: Unit
    denominator_value: float
    denominator_unit: Unit

    @override
    def get_unquantified(self) -> LiquidConcentration | GaseousPercentage:
        # Gas percentage is a special case
        if self.numerator_unit.identifier == ConceptId(PERCENT_CONCEPT_ID):
            # Same without denominator
            return GaseousPercentage(
                numerator_value=self.numerator_value,
                numerator_unit=self.numerator_unit,
            )

        return LiquidConcentration(
            # Divide to get the concentration
            numerator_value=self.numerator_value / self.denominator_value,
            numerator_unit=self.numerator_unit,
            denominator_unit=self.denominator_unit,
        )

    @override
    def _unit_matches(self, other: _StrengthMeta) -> bool:
        concentration = self.get_unquantified()
        return concentration._unit_matches(other)  # pyright: ignore[reportPrivateUsage]  # noqa: E501

    @override
    def _values_match(self, other: _StrengthMeta):
        if isinstance(other, LiquidQuantity):
            # Check total volume first
            diff = self.denominator_value / other.denominator_value
            if not (_LOW <= diff <= _HIGH):
                return False

        return self.get_unquantified()._values_match(other)  # pyright: ignore[reportPrivateUsage]  # noqa: E501

    def __post_init__(self):
        # Inherit checks from LiquidConcentration, as we are implicitly
        # a subclass
        LiquidConcentration.__post_init__(self)  # pyright: ignore[reportArgumentType]  # noqa: E501

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


# Shorthand for unquantified strength types
type UnquantifiedStrength = (
    SolidStrength | LiquidConcentration | GaseousPercentage
)

# Exhaustive list of strength types
type Strength = UnquantifiedStrength | LiquidQuantity


# Derived concepts
# # RxNorm
@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrugComponent[Id: ConceptIdentifier, S: UnquantifiedStrength](
    DrugNode[Id]
):
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

    @override
    def get_precise_ingredients(self) -> list[PreciseIngredient | None]:
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
class BrandedDrugComponent[Id: ConceptIdentifier, S: UnquantifiedStrength](
    _MulticomponentMixin[Id, S], DrugNode[Id]
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

    @override
    def get_strength_data(self) -> SortedTuple[BoundStrength[Id, S]]:
        # Return contents of unwrapped tuples
        return SortedTuple([
            cdc.get_strength_data()[0] for cdc in self.clinical_drug_components
        ])

    @override
    def get_precise_ingredients(self) -> list[PreciseIngredient | None]:
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
    def get_brand_name(self) -> BrandName[Id]:
        return self.brand_name


@dataclass(frozen=True, order=True, eq=True, slots=True)
class ClinicalDrugForm[Id: ConceptIdentifier](DrugNode[Id]):
    identifier: Id
    dose_form: DoseForm[Id]
    ingredients: SortedTuple[Ingredient[Id]]

    @override
    def get_dose_form(self) -> DoseForm[Id]:
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
    brand_name: BrandName[Id]

    @override
    def get_dose_form(self) -> DoseForm[Id]:
        return self.clinical_drug_form.dose_form

    @override
    def get_brand_name(self) -> BrandName[Id]:
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
class ClinicalDrug[Id: ConceptIdentifier, S: UnquantifiedStrength](
    _MulticomponentMixin[Id, S], DrugNode[Id]
):
    identifier: Id
    form: ClinicalDrugForm[Id]
    contents: SortedTuple[BoundStrength[Id, S]]
    # Those are the precise ingredients inherited from the components, if any
    precise_ingredients: tuple[PreciseIngredient | None]

    @override
    def get_dose_form(self) -> DoseForm[Id]:
        return self.form.dose_form

    @override
    def get_precise_ingredients(self) -> tuple[PreciseIngredient | None]:
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
class BrandedDrug[Id: ConceptIdentifier, S: UnquantifiedStrength](DrugNode[Id]):
    # TODO: Check if dosage may be different from CD counterpart
    identifier: Id
    clinical_drug: ClinicalDrug[Id, S]
    brand_name: BrandName[Id]

    @override
    def get_dose_form(self) -> DoseForm[Id]:
        return self.clinical_drug.form.dose_form

    @override
    def get_brand_name(self) -> BrandName[Id]:
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
    def get_precise_ingredients(self) -> tuple[PreciseIngredient | None]:
        return self.clinical_drug.get_precise_ingredients()


# Quantified liquid forms
type _Concentration = LiquidConcentration | GaseousPercentage


@dataclass(frozen=True, order=True, eq=True, slots=True)
class QuantifiedClinicalDrug[Id: ConceptIdentifier, C: _Concentration](
    DrugNode[Id,]
):
    identifier: Id
    contents: SortedTuple[BoundStrength[Id, LiquidQuantity]]
    unquantified: ClinicalDrug[Id, C]

    @override
    def get_dose_form(self) -> DoseForm[Id]:
        return self.unquantified.form.dose_form

    @override
    def get_precise_ingredients(self) -> tuple[PreciseIngredient | None]:
        return self.unquantified.get_precise_ingredients()

    def __post_init__(self):
        # Check strength data consistency
        if (
            len({
                (quantity.denominator_value, quantity.denominator_unit)
                for _, quantity in self.contents
            })
            > 1
        ):
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
    ) -> SortedTuple[BoundStrength[Id, LiquidQuantity]]:
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
        if not isinstance(other_data[0][1], LiquidQuantity):
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
        if not (_LOW <= diff <= _HIGH) or own_u != other_u:
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
    brand_name: BrandName[Id]
    # Redundant, hierarchy builder should check for ancestor consistency
    # unquantified: BrandedDrug[Id, C]
    # contents: SortedTuple[BoundStrength[Id, LiquidQuantity]]

    @override
    def get_dose_form(self) -> DoseForm[Id]:
        return self.unbranded.get_dose_form()

    @override
    def get_precise_ingredients(self) -> tuple[PreciseIngredient | None]:
        return self.unbranded.get_precise_ingredients()

    @override
    def get_brand_name(self) -> BrandName[Id]:
        return self.brand_name

    @override
    def get_strength_data(
        self,
    ) -> SortedTuple[BoundStrength[Id, LiquidQuantity]]:
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
