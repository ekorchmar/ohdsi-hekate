"""
Contains the ForeignDrugNode class, which represents an unknown node in the
drug concept hierarchy.
"""

from dataclasses import dataclass
from typing import NoReturn, override, NamedTuple
from rx_model.drug_classes.generic import (
    ConceptCodeVocab,
    ConceptId,
    ConceptIdentifier,
    BoundStrength,
    DrugNode,
)
import rx_model.drug_classes.atom as a
import rx_model.drug_classes.strength as st
import rx_model.drug_classes.complex as c
from rx_model.descriptive import StrengthConfiguration

from utils.classes import SortedTuple, PyRealNumber
from utils.enums import ConceptClassId
from utils.constants import BOX_SIZE_LIMIT
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
    | c.ClinicalDrugBox[Id, st.UnquantifiedStrength]
    | c.BrandedDrugBox[Id, st.UnquantifiedStrength]
    | c.QuantifiedClinicalDrug[Id, st.LiquidConcentration | st.GasPercentage]
    | c.QuantifiedBrandedDrug[Id]
    | c.QuantifiedClinicalBox[Id, st.LiquidConcentration | st.GasPercentage]
    | c.QuantifiedBrandedBox[Id, st.LiquidConcentration | st.GasPercentage]
)

# PseudoUnit is verbatim string representation of a unit in source data
type PseudoUnit = str
type BoundForeignStrength = tuple[
    a.Ingredient[ConceptCodeVocab], ForeignStrength | None
]


class ForeignStrength(NamedTuple):
    """
    Represents a strength entry in a foreign node.

    It is purposefully detached from the ingredient information, as the logic
    for handling mapping information is processed separately.
    """

    amount_value: PyRealNumber | None
    amount_unit: PseudoUnit | None
    numerator_value: PyRealNumber | None
    numerator_unit: PseudoUnit | None
    denominator_value: PyRealNumber | None
    denominator_unit: PseudoUnit | None

    def derive_configuration(self) -> StrengthConfiguration:
        """
        Fallible way of deriving the strength configuration from the strength
        data. Implicitly assumes that the strength data is valid, as it is not
        the concern of this class.
        """
        if self.amount_unit is not None:
            return StrengthConfiguration.AMOUNT_ONLY
        if self.denominator_value is not None:
            return StrengthConfiguration.LIQUID_QUANTITY
        if self.numerator_unit == "%":
            return StrengthConfiguration.GAS_PERCENTAGE
        return StrengthConfiguration.LIQUID_CONCENTRATION


class ForeignNodePrototype(NamedTuple):
    """
    Represents a data pack prototype of a foreign node, with all the information
    required to create a ForeignDrugNode instance.

    The prototype contents are derived from the source data and use definitions
    native to the source data. It is job of a translator to convert these
    definitions to valid ForeignDrugNode instances.
    """

    identifier: ConceptCodeVocab
    strength_data: SortedTuple[BoundForeignStrength]
    brand_name: a.BrandName[ConceptCodeVocab] | None = None
    dose_form: a.DoseForm[ConceptCodeVocab] | None = None
    supplier: a.Supplier[ConceptCodeVocab] | None = None
    box_size: int | None = None

    # WARN: precise_ingredients are not specifiable in the source for now


class PrecedenceData(NamedTuple):
    """
    Metadata package that represents the precedence data for a foreign node,
    which is used to disambiguate the best result when multiple nodes are
    matched.

    Each value of the tuple represents the precedence ordering of the specific
    attribute, with lower values indicating higher precedence.
    """

    ingredient_diff: int
    dose_form_diff: int = 0
    brand_name_diff: int = 0
    supplier_diff: int = 0


@dataclass(frozen=True, slots=True)
class ForeignDrugNode[S: st.Strength | None](DrugNode[ConceptId, S]):
    """
    Represents an unknown node in the drug concept hierarchy. This is used to
    model a virtual representation of a source drug concept in form native to
    RxNorm/RxNorm-Extension model.
    """

    # Metadata populated at creation
    precedence_data: PrecedenceData
    identifier: ConceptId

    strength_data: SortedTuple[BoundStrength[ConceptId, S]]
    dose_form: a.DoseForm[ConceptId] | None = None
    brand_name: a.BrandName[ConceptId] | None = None
    supplier: a.Supplier[ConceptId] | None = None
    box_size: int | None = None

    # Is curently None for practical purposes
    precise_ingredients: list[a.PreciseIngredient | None] | None = None

    @override
    def is_superclass_of(
        self,
        other: DrugNode[ConceptId, st.Strength | None],
        passed_hierarchy_checks: bool = True,
    ) -> NoReturn:
        del passed_hierarchy_checks, other
        raise NotImplementedError(
            "Cannot check superclass relationship with a foreign node."
        )

    @override
    def get_strength_data(
        self,
    ) -> SortedTuple[BoundStrength[ConceptId, S]]:
        return self.strength_data

    @override
    def get_precise_ingredients(
        self,
    ) -> list[a.PreciseIngredient | None]:
        # If no precise ingredients are provided, return a list of Nones
        if self.precise_ingredients is None:
            return [None] * len(self.strength_data)
        return self.precise_ingredients

    @override
    def get_brand_name(self) -> a.BrandName[ConceptId] | None:
        return self.brand_name

    @override
    def get_dose_form(self) -> a.DoseForm[ConceptId] | None:
        return self.dose_form

    @override
    def get_supplier(self) -> a.Supplier[ConceptId] | None:
        return self.supplier

    @override
    def get_box_size(self) -> int | None:
        return self.box_size

    def __post_init__(self):
        self.validate_strength_data()
        self.validate_precise_ingredients()
        self.validate_precedence_data()
        self.forbid_formless_quantities()
        self.validate_box_size()
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

    def best_case_class(self) -> type[_AnyComplex[ConceptId]]:
        """
        Tries to infer the target class of this foreign node based on the
        presence of attributes and shape of the strength data.
        """

        box_size = self.box_size is not None
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
        elif box_size:
            if isinstance(strength, st.LiquidQuantity):
                # QCB or QBB
                return (
                    c.QuantifiedBrandedDrug
                    if branded
                    else c.QuantifiedClinicalDrug
                )
            # CB or BB
            return c.BrandedDrugBox if branded else c.ClinicalDrugBox
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

    def validate_precedence_data(self):
        """
        Ensures that the precedence data is valid.
        """
        if any(p < 0 for p in self.precedence_data):
            raise ForeignNodeCreationError(
                f"Precedence values must be non-negative, but Node "
                f"{self.identifier} has: {self.precedence_data}."
            )

        # Non-zero precedence values for unset attributes are indicative of
        # things going very wrong
        pd = self.precedence_data
        for name, attr, prc in (
            ("dose form", self.dose_form, pd.dose_form_diff),
            ("brand name", self.brand_name, pd.brand_name_diff),
            ("supplier", self.supplier, pd.supplier_diff),
        ):
            if attr is None and prc != 0:
                raise ForeignNodeCreationError(
                    f"Precedence value for unset attribute {name} must be 0, "
                    f"but Node {self.identifier} has: {prc}."
                )

    def validate_box_size(self):
        """
        Ensures that the box size is valid.
        """
        if self.box_size is None:
            return

        if BOX_SIZE_LIMIT < self.box_size <= 0:
            raise ForeignNodeCreationError(
                f"Box size of Foreign Node {self.identifier} must "
                f"be a positive integer less than or equal to "
                f"{BOX_SIZE_LIMIT}, not {self.box_size}."
            )

        if self.strength_data[0][1] is None:
            raise ForeignNodeCreationError(
                f"Box size can only be set for known strength data, but "
                f"Node {self.identifier} has no dosage and {self.box_size} "
                f"as the box size."
            )

        if self.dose_form is None:
            raise ForeignNodeCreationError(
                f"Box size can only be set for nodes with a dose form, but "
                f"Node {self.identifier} has no form and {self.box_size} "
                f"as the box size."
            )

    @override
    @classmethod
    def from_definitions(
        cls,
        identifier: ConceptId,
        parents: dict[
            ConceptClassId, list[DrugNode[ConceptId, st.Strength | None]]
        ],
        attributes: dict[
            ConceptClassId,
            a.BrandName[ConceptId]
            | a.DoseForm[ConceptId]
            | a.Supplier[ConceptId],
        ],
        precise_ingredients: list[a.PreciseIngredient],
        strength_data: SortedTuple[BoundStrength[ConceptId, S]],
        box_size: int | None,
    ) -> NoReturn:
        raise NotImplementedError(
            "Foreign nodes cannot be created from definitions."
        )
