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
from utils.exceptions import (
    ForeignNodeCreationError,
    InvalidConceptIdError,
    UnmappedSourceConceptError,
)

type _AtomLookupCallable = Callable[[ConceptId], AtomicConcept[ConceptId]]


class NodeTranslator:
    """
    Maps atomic drug attributes to their corresponding RxNorm concepts and
    translates a source drug node definition into permutations of RxNorm-native
    drug nodes.
    """

    def __init__(
        self,
        rx_atoms: Atoms[ConceptId],
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
        self._get_atom: _AtomLookupCallable = rx_atoms.lookup_unknown
        self._unit_storage: Mapping[ConceptId, Unit] = rx_atoms.unit

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
        Translate a single strength measure (a pair of value and unit) to a
        sequence of possible value-unit pairs in RxNorm-native units.
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

        This produces all possible node variations in order of precedence. These
        nodes are known as "virtual nodes", and should only exist until
        evaluated. They must be then disambiguated by heuristic comparison of
        mapping results.
        """
        self.logger.debug(f"Getting permutations for node: {node}")

        # Order of preference is: ingredient, dose form, brand name, supplier
        node_ingredients: list[Ingredient] = []
        strengths: list[S] = []
        for ingredient, strength in node.strength_data:
            node_ingredients.append(ingredient)
            strengths.append(strength)

        attribute_mappings = []
        for node_attribute in [
            *node_ingredients,
            node.dose_form,
            node.brand_name,
            node.supplier,
        ]:
            if node_attribute is not None:
                try:
                    mappings = self._concept_map[node_attribute.identifier]
                except KeyError:
                    # NOTE: Not necessarily an error: maybe a new ingredient has
                    # to be created from source.
                    # TODO: track these cases for the reporting and hierarchy
                    # extension
                    raise UnmappedSourceConceptError(
                        f"Unmapped attribute: {node_attribute}"
                    )
                else:
                    attribute_mappings.append(mappings)
            else:
                attribute_mappings.append([None])

        attribute_permutations = product(*attribute_mappings)

        # Share the identifier across all permutations
        shared_concept_id = concept_id_factory()
        any_creations_succeeded = False
        for *ings, df_id, bn_id, sp_id in attribute_permutations:
            strength_data: list[tuple[Ingredient[ConceptId], S]] = []

            for ing, stgh in zip(ings, strengths):
                ing_object = self._try_get_atom(ing, Ingredient)
                assert isinstance(ing_object, Ingredient)
                strength_data.append((ing_object, stgh))

            attrs = {
                "dose_form": (DoseForm, df_id),
                "brand_name": (BrandName, bn_id),
                "supplier": (Supplier, sp_id),
            }
            fn_attrs = {}
            for attr, (class_, identifier) in attrs.items():
                if identifier is None:
                    continue
                fn_attrs[attr] = self._try_get_atom(identifier, class_)

            try:
                fn = ForeignDrugNode(
                    identifier=shared_concept_id,
                    strength_data=SortedTuple(strength_data),
                    # We explicitly checked that these are the correct types
                    **fn_attrs,
                )
                any_creations_succeeded = True
                yield fn

            except ForeignNodeCreationError as e:
                # NOTE:
                # Most likely to happen if the different source ingredients end
                # up having the same RxNorm concept ID through precedence
                # mapping. This actually could be handled differently by simply
                # summing the strengths, but that needs a consensus from the
                # Vocabulary team first.
                self.logger.debug(
                    f"Failed to create node: {e}. Skipping this permutation."
                )
                # In any case, this should not re-raise, unless no successful
                # nodes are created
        if not any_creations_succeeded:
            raise ForeignNodeCreationError(
                "No successful node creations from permutations. Check the "
                "ingredient precedence mappings."
            )

    def _try_get_atom(
        self,
        identifier: ConceptId,
        expected_class: type[AtomicConcept[ConceptId]],
    ) -> AtomicConcept[ConceptId]:
        try:
            value = self._get_atom(identifier)
        except InvalidConceptIdError as e:
            self.logger.error(
                f"Tried getting {expected_class.__name__} "
                f"by {identifier} which is not a valid attribute."
            )
            raise UnmappedSourceConceptError(
                f"Invalid attribute: {identifier} for {expected_class.__name__}"
            ) from e

        if value is not None and not isinstance(value, expected_class):
            self.logger.error(
                f"Expected {expected_class.__name__}, got {value}"
            )
            raise UnmappedSourceConceptError(
                f"Invalid attribute: {value} for {expected_class.__name__}"
            )
        return value
