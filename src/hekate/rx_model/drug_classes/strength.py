import math  # For NaN checks
from abc import (  # For abstract interfaces
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import override  # For type annotations

from rx_model.drug_classes import atom as a  # For unit classes
from rx_model.drug_classes.generic import ConceptId
from utils.constants import (
    PERCENT_CONCEPT_ID,  # For gaseous percentage check
)
from utils.constants import STRENGTH_CLOSURE_BOUNDARY_HIGH
from utils.constants import STRENGTH_CLOSURE_BOUNDARY_LOW
from utils.exceptions import RxConceptCreationError

HIGH = STRENGTH_CLOSURE_BOUNDARY_HIGH
LOW = STRENGTH_CLOSURE_BOUNDARY_LOW


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


@dataclass(frozen=True, order=True, eq=True, slots=True)
class SolidStrength(_StrengthMeta):
    """
    Single value/unit combination for dosage information.
    """

    amount_value: float
    amount_unit: a.Unit

    def __post_init__(self):
        if self.amount_value < 0:
            raise RxConceptCreationError(
                f"Solid strength must have a non-negative value, not {
                    self.amount_value
                }."
            )

        if math.isnan(self.amount_value):
            raise RxConceptCreationError(
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
        return LOW <= diff <= HIGH


@dataclass(frozen=True, order=True, eq=True, slots=True)
class LiquidConcentration(_StrengthMeta):
    """
    Dosage given as unquantified concentration.
    """

    numerator_value: float
    numerator_unit: a.Unit
    # Null for denominator value
    denominator_unit: a.Unit

    def __post_init__(self):
        if self.numerator_value <= 0:
            raise RxConceptCreationError(
                f"Liquid concentration must have a positive numerator "
                f"value, not {self.numerator_value}."
            )

        if math.isnan(self.numerator_value):
            raise RxConceptCreationError(
                f"Liquid concentration must have a numeric numerator value, "
                f"not {self.numerator_value}."
            )

    @override
    def get_unquantified(self) -> "LiquidConcentration":
        return self

    @override
    def _unit_matches(self, other: _StrengthMeta) -> bool:
        if not isinstance(unq := other.get_unquantified(), LiquidConcentration):
            return False

        assert isinstance(unq, LiquidConcentration)
        return (self.numerator_unit, self.denominator_unit) == (
            unq.numerator_unit,
            unq.denominator_unit,
        )

    @override
    def _values_match(self, other: _StrengthMeta) -> bool:
        other = other.get_unquantified()
        assert isinstance(other, LiquidConcentration)
        diff = self.numerator_value / other.numerator_value
        return LOW <= diff <= HIGH


@dataclass(frozen=True, order=True, eq=True, slots=True)
class GasPercentage(_StrengthMeta):
    """
    Special case of concentration for gases. Only specifies numerator value
    and unit.
    """

    numerator_value: float
    numerator_unit: a.Unit
    # Null for denominator value
    # Null for denominator unit

    def __post_init__(self):
        # Numerator unit must be a percentage
        if self.numerator_unit.identifier != ConceptId(PERCENT_CONCEPT_ID):
            raise RxConceptCreationError(
                f"Gaseous concentration must be percent ({PERCENT_CONCEPT_ID})"
                f"percentage unit, not {self.numerator_unit}."
            )

        # Numerator value must be a positive number < 100
        if 0 <= self.numerator_value > 100:
            raise RxConceptCreationError(
                f"Gaseous concentration must have a positive numerator "
                f"value not exceeding 100, not {self.numerator_value}."
            )

    @override
    def get_unquantified(self) -> "GasPercentage":
        return self

    @override
    def _unit_matches(self, other: _StrengthMeta) -> bool:
        # There is only one valid unit for gaseous percentages
        return isinstance(other.get_unquantified(), GasPercentage)

    @override
    def _values_match(self, other: _StrengthMeta) -> bool:
        other = other.get_unquantified()
        assert isinstance(other, GasPercentage)
        diff = self.numerator_value / other.numerator_value
        return LOW <= diff <= HIGH


@dataclass(frozen=True, order=True, eq=True, slots=True)
class LiquidQuantity(_StrengthMeta):
    """
    Quantified liquid or gaseous dosage with both total content and volume.
    """

    # NOTE: we are an implicit subclass of LiquidConcentration and
    # GaseousPercentage, so we allow accessing their protected methods here.

    numerator_value: float
    numerator_unit: a.Unit
    denominator_value: float
    denominator_unit: a.Unit

    @override
    def get_unquantified(self) -> LiquidConcentration | GasPercentage:
        # Gas percentage is a special case
        if self.numerator_unit.identifier == ConceptId(PERCENT_CONCEPT_ID):
            # Same without denominator
            return GasPercentage(
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
            if not (LOW <= diff <= HIGH):
                return False

        return self.get_unquantified()._values_match(other)  # pyright: ignore[reportPrivateUsage]  # noqa: E501

    def denominator_matches(self, other: "LiquidQuantity") -> bool:
        if self.denominator_unit != other.denominator_unit:
            return False

        diff = self.denominator_value / other.denominator_value
        return LOW <= diff <= HIGH

    def __post_init__(self):
        # Inherit checks from LiquidConcentration, as we are implicitly
        # a subclass
        LiquidConcentration.__post_init__(self)  # pyright: ignore[reportArgumentType]  # noqa: E501

        if self.denominator_value <= 0:
            raise RxConceptCreationError(
                f"Liquid quantity must have a positive denominator value, not "
                f"{self.denominator_value}."
            )

        if math.isnan(self.denominator_value):
            raise RxConceptCreationError(
                f"Liquid quantity must have a numeric denominator value, not "
                f"{self.denominator_value}."
            )

        if (
            self.numerator_unit == self.denominator_unit
            and self.numerator_value >= self.denominator_value
        ):
            raise RxConceptCreationError(
                f"Liquid quantity must have a numerator value less than the "
                f"denominator value, not {self.numerator_value} >= "
                f"{self.denominator_value}."
            )


# Shorthand for unquantified strength types
type UnquantifiedStrength = SolidStrength | LiquidConcentration | GasPercentage

# Exhaustive list of strength types
type Strength = UnquantifiedStrength | LiquidQuantity
