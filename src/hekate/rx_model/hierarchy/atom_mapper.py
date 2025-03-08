"""
Contains the AtomMapper class, which is used to map atomic drug attributes to
their corresponding RxNorm concepts.
"""

import logging
from collections import OrderedDict
from collections.abc import Generator, Mapping, Sequence
from itertools import product
from typing import Callable

import polars as pl
from rx_model.drug_classes import (
    BrandName,
    ConceptCodeVocab,
    ConceptId,
    DoseForm,
    ForeignDrugNode,
    Ingredient,
    Strength,
    Supplier,
    Unit,
)
from rx_model.hierarchy.generic import AtomicConcept, PseudoUnit
from rx_model.hierarchy.hosts import Atoms
from utils.classes import SortedTuple

type AtomLookupCallable = Callable[[ConceptId], AtomicConcept[ConceptId]]


class AtomMapper:
    """
    Maps atomic drug attributes to their corresponding RxNorm concepts.
    """

    def __init__(
        self,
        source_atoms: Atoms,
        logger: logging.Logger,
    ):
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info("Initializing AtomMapper")

        # Maps atomic attributes to their corresponding RxNorm concepts
        # in order of precedence
        self._concept_map: dict[ConceptCodeVocab, Sequence[ConceptId]] = {}
        # Maps pseudo-units to their corresponding RxNorm units in order of
        # precedence and with preserved conversion factor
        self._unit_map: dict[PseudoUnit, OrderedDict[Unit, float]] = {}

        # External references to the storage dictionaries
        self._get_atom: AtomLookupCallable = source_atoms.lookup_unknown
        self._unit_storage: Mapping[ConceptId, Unit] = source_atoms.unit

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

    def populate_from_frame(
        self, frame: pl.DataFrame, pseudo_units: list[PseudoUnit]
    ):
        """
        Populate the mappings from a DataFrame.
        """
        frame = frame.select(
            concept_code="concept_code_1",
            vocabulary_id="vocabulary_id_1",
            concept_id="concept_id_2",
            precedence=pl.coalesce(
                pl.col("precedence"), pl.lit(1, dtype=pl.UInt8)
            ),
            conversion_factor="conversion_factor",
            # NOTE: Order of precedence is supposedly preserved within
            # each group
            # https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by.html
        ).sort("precedence")

        atom_frame = (
            frame.filter(
                ~pl.col("concept_code").is_in(pseudo_units),
            )
            .select("concept_code", "vocabulary_id", "concept_id")
            .group_by("concept_code", "vocabulary_id")
            .all()
            .select("concept_code", "vocabulary_id", concept_ids="concept_id")
        )

        self.logger.info(
            f"Adding mappings for attribites of {len(atom_frame)} atoms"
        )

        unit_frame = (
            frame.filter(
                pl.col("concept_code").is_in(pseudo_units),
            )
            .select("concept_code", "concept_id", "conversion_factor")
            .group_by("concept_code")
            .all()
            .select(
                "concept_code",
                concept_ids="concept_id",
                conversion_factors="conversion_factor",
            )
        )

        self.logger.info(f"Adding mappings for {len(unit_frame)} units")

        for row in atom_frame.iter_rows():
            id = ConceptCodeVocab(
                concept_code=row[0],
                vocabulary_id=row[1],
            )
            concept_ids = row[2]
            self.add_concept_mappings(id, list(map(ConceptId, concept_ids)))

        for row in unit_frame.iter_rows():
            pseudo_unit: PseudoUnit = row[0]
            concept_ids: list[int] = row[1]
            conversion_factors: list[float] = row[2]
            unit_map: OrderedDict[Unit, float] = OrderedDict()
            for cid, factor in zip(concept_ids, conversion_factors):
                unit: Unit = self._unit_storage[ConceptId(cid)]
                unit_map[unit] = factor

            self.add_unit_mappings(pseudo_unit, unit_map)

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
        self.logger.debug(f"Getting permutations for node: {node}")

        # Order of preference is: ingredient, dose form, brand name, supplier
        ingredient_codes: list[ConceptCodeVocab] = []
        strengths: list[S] = []
        for ingredient, strength in node.strength_data:
            if ingredient.identifier not in self._concept_map:
                raise ValueError(f"Unknown ingredient: {ingredient}")
            ingredient_codes.append(ingredient.identifier)
            strengths.append(strength)

        M = self._concept_map
        attribute_permutations = product(
            *[M[ingredient_code] for ingredient_code in ingredient_codes],
            M[node.dose_form.identifier] if node.dose_form else [None],
            M[node.brand_name.identifier] if node.brand_name else [None],
            M[node.supplier.identifier] if node.supplier else [None],
        )

        for *ings, df, bn, sp in attribute_permutations:
            strength_data: list[tuple[Ingredient[ConceptId], S]] = []

            for ing, stgh in zip(ings, strengths):
                assert ing is not None
                ing_object = self._get_atom(ing)
                assert isinstance(ing_object, Ingredient)
                strength_data.append((ing_object, stgh))

            dose_form = brand_name = supplier = None
            if df:
                dose_form = self._get_atom(df)
            if bn:
                brand_name = self._get_atom(bn)
            if sp:
                supplier = self._get_atom(sp)

            for attr, class_ in [
                (dose_form, DoseForm),
                (brand_name, BrandName),
                (supplier, Supplier),
            ]:
                if attr is not None and not isinstance(attr, class_):
                    raise ValueError(f"Expected {class_.__name__}, got {attr}")

            yield ForeignDrugNode(
                identifier=ConceptId(concept_id_factory()),
                strength_data=SortedTuple(strength_data),
                # We just explicitly checked that these are the correct types
                dose_form=dose_form,  # pyright: ignore[reportArgumentType]
                brand_name=brand_name,  # pyright: ignore[reportArgumentType]
                supplier=supplier,  # pyright: ignore[reportArgumentType]
            )
