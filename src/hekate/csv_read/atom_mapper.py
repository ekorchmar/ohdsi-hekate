"""
Contains the AtomMapper class, which is used to map atomic drug attributes to
their corresponding RxNorm concepts.
"""

from collections.abc import Sequence, Generator, Mapping
from hekate.rx_model.drug_classes.strength import Strength
from hekate.utils.classes import SortedTuple
from rx_model.drug_classes import ForeignDrugNode
from rx_model.drug_classes import (
    ConceptId,
    ConceptCodeVocab,
    Unit,
    Ingredient,
    BrandName,
    DoseForm,
    Supplier,
)
from rx_model.hierarchy.generic import AtomicConcept, PseudoUnit
from collections import OrderedDict
from typing import Callable
from itertools import product


class AtomMapper:
    """
    Maps atomic drug attributes to their corresponding RxNorm concepts.
    """

    def __init__(
        self,
        atom_storage: dict[ConceptId, AtomicConcept[ConceptId]],
        unit_storage: dict[ConceptId, Unit],
    ):
        # Maps atomic attributes to their corresponding RxNorm concepts
        # in order of precedence
        self._concept_map: dict[ConceptCodeVocab, Sequence[ConceptId]] = {}
        # Maps pseudo-units to their corresponding RxNorm units in order of
        # precedence and with preserved conversion factor
        self._unit_map: dict[PseudoUnit, OrderedDict[Unit, float]] = {}

        # External references to the storage dictionaries
        self._atom_storage: Mapping[ConceptId, AtomicConcept[ConceptId]] = (
            atom_storage
        )
        self._unit_storage: Mapping[ConceptId, Unit] = unit_storage

    def add_concept_mappings(
        self, concept_vocab: ConceptCodeVocab, concept_ids: Sequence[ConceptId]
    ):
        """
        Add mappings from a concept vocabulary to a sequence of concept IDs.
        """
        self._concept_map[concept_vocab] = concept_ids

    def add_unit_mappings(
        self, pseudo_unit: PseudoUnit, unit_map: OrderedDict[Unit, float]
    ):
        """
        Add mappings from a pseudo-unit to a sequence of concept IDs with
        conversion factors.
        """
        self._unit_map[pseudo_unit] = unit_map

    def translate_strength_measure(
        self, value: float, unit: PseudoUnit
    ) -> Generator[tuple[float, Unit], None, None]:
        """
        Translate a strength measure to a sequence of possible value-unit pairs.
        """

        if unit not in self._unit_map:
            raise ValueError(f"Unknown source unit: {unit}")

        for rx_unit, conversion_factor in self._unit_map[unit].items():
            yield value * conversion_factor, rx_unit

    def translate_node[S: Strength | None](
        self,
        node: ForeignDrugNode[ConceptCodeVocab, S],
        concept_id_factory: Callable[[], ConceptId],
    ) -> Generator[ForeignDrugNode[ConceptId, S], None, None]:
        """
        Translate a source drug node definitions int a sequence of RxNorm-native
        drug node definitions.

        This produces all possible node variations in order of precedence.
        """

        # Order of preference is: ingredient, dose form, brand name, supplier
        ingredient_codes: list[ConceptCodeVocab] = []
        strengths: list[S] = []
        for ingredient, strength in node.strength_data:
            if ingredient not in self._atom_storage:
                raise ValueError(f"Unknown ingredient: {ingredient}")
            ingredient_codes.append(ingredient.identifier)
            strengths.append(strength)

        M = self._concept_map
        attribute_permutations = product(
            *[M[ingredient] for ingredient in ingredient_codes],
            M[node.dose_form.identifier] if node.dose_form else [None],
            M[node.brand_name.identifier] if node.brand_name else [None],
            M[node.supplier.identifier] if node.supplier else [None],
        )
        for *ings, df, bn, sp in attribute_permutations:
            strength_data: list[tuple[Ingredient[ConceptId], S]] = []

            for ing, stgh in zip(ings, strengths):
                assert ing is not None
                ing_object = self._atom_storage[ing]
                assert isinstance(ing_object, Ingredient)
                strength_data.append((ing_object, stgh))

            df = bn = sp = None
            if df:
                dose_form = self._atom_storage[df]
                assert isinstance(dose_form, DoseForm)
            if bn:
                brand_name = self._atom_storage[bn]
                assert isinstance(brand_name, BrandName)
            if sp:
                supplier = self._atom_storage[sp]
                assert isinstance(supplier, Supplier)

            yield ForeignDrugNode(
                identifier=ConceptId(concept_id_factory()),
                strength_data=SortedTuple(strength_data),
                dose_form=df,
                brand_name=bn,
                supplier=sp,
            )
