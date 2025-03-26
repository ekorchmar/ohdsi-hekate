"""
Contains the NodeTranslator class, which is used to produce virtual nodes
from source drug node definitions.
"""

import logging
from collections import OrderedDict
from collections.abc import Generator, Mapping, Sequence
from itertools import product
from typing import Callable, NamedTuple

import polars as pl
from csv_read.source_input import BuildRxEInput
from rx_model import drug_classes as dc
from rx_model.hierarchy.generic import AtomicConcept
from rx_model.hierarchy.hosts import Atoms
from utils.classes import SortedTuple, PyRealNumber
from utils.constants import StrengthConfiguration
from utils.exceptions import (
    ForeignNodeCreationError,
    InvalidConceptIdError,
    UnmappedSourceConceptError,
)

type _AtomLookupCallable = Callable[[dc.ConceptId], AtomicConcept[dc.ConceptId]]
type _Attribute[Id: dc.ConceptIdentifier] = (
    dc.DoseForm[Id] | dc.BrandName[Id] | dc.Supplier[Id] | dc.Ingredient[Id]
)
type _PrecedentedTarget[A: _Attribute[dc.ConceptId]] = tuple[int, A | None]
type _PrecedentedIngredient = tuple[int, dc.Ingredient[dc.ConceptId]]
type _PrecedentedDoseForm = _PrecedentedTarget[dc.DoseForm[dc.ConceptId]]
type _PrecedentedBrandName = _PrecedentedTarget[dc.BrandName[dc.ConceptId]]
type _PrecedentedSupplier = _PrecedentedTarget[dc.Supplier[dc.ConceptId]]

# NOTE: Precedence of bound strength is determined only by the ingredient. Unit
# mapping precedence is meaningless for disambiguation.
type _PrecedentedBoundStrength[S: dc.Strength | None] = tuple[
    int, dc.BoundStrength[dc.ConceptId, S]
]
# NOTE: Precedence of strength data is determined by the sum of all contributing
# components' precedences -- meaning the sum of all ingredient precedences.
type _PrecedentedStrengthData[S: dc.Strength | None] = tuple[
    int, SortedTuple[dc.BoundStrength[dc.ConceptId, S]]
]


class AttributePermutations(NamedTuple):
    dose_form: list[_PrecedentedDoseForm]
    brand_name: list[_PrecedentedBrandName]
    supplier: list[_PrecedentedSupplier]


class NodeTranslator:
    """
    Maps atomic drug attributes to their corresponding RxNorm concepts and
    translates a source drug node definition into permutations of RxNorm-native
    drug nodes.
    """

    STRENGTH_CLASS: dict[StrengthConfiguration, type[dc.Strength]] = {
        StrengthConfiguration.AMOUNT_ONLY: dc.SolidStrength,
        StrengthConfiguration.LIQUID_CONCENTRATION: dc.LiquidConcentration,
        StrengthConfiguration.LIQUID_QUANTITY: dc.LiquidQuantity,
        StrengthConfiguration.GAS_PERCENTAGE: dc.GasPercentage,
    }

    def __init__(
        self,
        rx_atoms: Atoms[dc.ConceptId],
        logger: logging.Logger,
    ):
        self.logger: logging.Logger = logger.getChild(self.__class__.__name__)
        self.logger.info("Initializing AtomMapper")

        # Maps atomic attributes to their corresponding RxNorm concepts
        # in order of precedence
        self._concept_map: dict[
            dc.ConceptCodeVocab, Sequence[dc.ConceptId]
        ] = {}
        # Maps pseudo-units to their corresponding RxNorm units in order of
        # precedence and with preserved conversion factor
        self._unit_map: dict[
            dc.PseudoUnit, OrderedDict[dc.Unit, PyRealNumber]
        ] = {}

        # External references to the storage dictionaries
        self._get_atom: _AtomLookupCallable = rx_atoms.lookup_unknown
        self._unit_storage: Mapping[dc.ConceptId, dc.Unit] = rx_atoms.unit

    def add_concept_mappings(
        self,
        concept_vocab: dc.ConceptCodeVocab,
        concept_ids: Sequence[dc.ConceptId],
    ):
        """
        Add mappings from a concept vocabulary to a sequence of concept IDs.
        """
        self._concept_map[concept_vocab] = concept_ids

    def add_unit_mappings(
        self,
        pseudo_unit: dc.PseudoUnit,
        unit_map: OrderedDict[dc.Unit, PyRealNumber],
    ):
        """
        Add mappings from a pseudo-dc.unit to a sequence of concept IDs with
        conversion factors.
        """
        self._unit_map[pseudo_unit] = unit_map

    def read_translations(self, source: BuildRxEInput):
        """
        Populate the mappings from a DataFrame.
        """
        frame = (
            source.rtcs.collect()
            .select(
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
            )
            .sort("precedence")
        )

        atom_frame = (
            frame.filter(
                ~pl.col("concept_code").is_in(source.pseudo_units),
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
                pl.col("concept_code").is_in(source.pseudo_units),
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
            concept_code: str = row[0]
            vocab_id: str = row[1]
            id = dc.ConceptCodeVocab(
                concept_code=concept_code,
                vocabulary_id=vocab_id,
            )
            concept_ids = row[2]
            self.add_concept_mappings(id, list(map(dc.ConceptId, concept_ids)))

        for row in unit_frame.iter_rows():
            pseudo_unit: dc.PseudoUnit = row[0]
            concept_ids: list[int] = row[1]
            conversion_factors: list[PyRealNumber] = row[2]
            unit_map: OrderedDict[dc.Unit, PyRealNumber] = OrderedDict()
            for cid, factor in zip(concept_ids, conversion_factors):
                unit: dc.Unit = self._unit_storage[dc.ConceptId(cid)]
                unit_map[unit] = factor

            self.add_unit_mappings(pseudo_unit, unit_map)

    def _translate_strength_measure(
        self, value: PyRealNumber, unit: dc.PseudoUnit
    ) -> Generator[tuple[PyRealNumber, dc.Unit], None, None]:
        """
        Translate a single strength measure (a pair of value and dc.unit) to a
        sequence of possible value-dc.unit pairs in RxNorm-native units.
        """

        if unit not in self._unit_map:
            raise ValueError(f"Unknown source dc.unit: {unit}")

        for rx_unit, conversion_factor in self._unit_map[unit].items():
            yield value * conversion_factor, rx_unit

    def _derive_strength_class(
        self, strength: dc.BoundForeignStrength
    ) -> type[dc.Strength] | None:
        """
        Derive the expected strength class from the strength data.

        Returns None if the strength data is also None.
        """
        if (foreign_strength := strength[1]) is None:
            return None
        expected = foreign_strength.derive_configuration()
        return self.STRENGTH_CLASS[expected]

    def translate_node(
        self,
        node_prototype: dc.ForeignNodePrototype,
        concept_id_factory: Callable[[], dc.ConceptId],
    ) -> Generator[dc.ForeignDrugNode[dc.Strength | None], None, None]:
        """
        Translate a source drug node definitions int a sequence of RxNorm-native
        drug node definitions.

        This produces all possible node variations in order of precedence. These
        nodes are known as "virtual nodes", and should only exist until
        evaluated. They must be then disambiguated by heuristic comparison of
        mapping results.
        """
        self.logger.debug(
            f"Getting permutations for node {node_prototype.identifier}"
        )
        # Share the identifier across all permutations
        shared_concept_id = concept_id_factory()

        dfs, bns, supps = self.get_attribute_permutations(
            node_prototype.dose_form,
            node_prototype.brand_name,
            node_prototype.supplier,
        )

        strength_data = node_prototype.strength_data

        # Redundant check
        strength_classes = {
            self._derive_strength_class(strength) for strength in strength_data
        }
        assert len(strength_classes) == 1, (
            f"Incorrect strength configurations in the strength data "
            f"for the node prototype {node_prototype.identifier}: total "
            f"of {len(strength_classes)}"
        )

        if (strength_shape := strength_classes.pop()) is None:
            strength_permutations = self._get_ingredient_permutations(
                ingredients=[ing for ing, _ in strength_data]
            )
        else:
            strength_permutations = self._get_strength_permutations(
                rows=strength_data,
                expected_class=strength_shape,
            )

        any_creations_succeeded = False
        combinations = product(
            strength_permutations,
            dfs,
            bns,
            supps,
        )
        for (p_s, st), (p_d, d), (p_b, b), (p_s, s) in combinations:
            precedence_data = dc.PrecedenceData(
                ingredient_diff=p_s,
                dose_form_diff=p_d,
                brand_name_diff=p_b,
                supplier_diff=p_s,
            )

            try:
                foreign_node = dc.ForeignDrugNode(
                    precedence_data=precedence_data,
                    strength_data=st,
                    identifier=shared_concept_id,
                    dose_form=d,
                    brand_name=b,
                    supplier=s,
                    box_size=node_prototype.box_size,  # Use as is
                )

            except ForeignNodeCreationError as e:
                # NOTE:
                # Most likely to happen if the different source ingredients end
                # up having the same RxNorm concept ID through precedence
                # mapping. This actually could be handled differently by simply
                # summing the strengths, but that needs a consensus from the
                # Vocabulary team first.
                self.logger.debug(
                    f"Failed to create node: {e}. Skipping this permutation. "
                    f"Precedence data: {precedence_data}"
                )
                # In any case, this should not re-raise, unless no successful
                # nodes are created
            else:
                any_creations_succeeded = True
                # NOTE: again the yield is covariant
                yield foreign_node

        if not any_creations_succeeded:
            raise ForeignNodeCreationError(
                f"No successful node created from permutations. for "
                f"{node_prototype.identifier}. Check the ingredient precedence "
                f"mappings."
            )

    def _try_get_atom[
        A: _Attribute[dc.ConceptId] | dc.Ingredient[dc.ConceptId]
    ](self, identifier: dc.ConceptId, expected_class: type[A]) -> A:
        """
        Try to get an atomic attribute by its identifier and check if it's of
        the expected class. If the attribute is not of the expected class, raise
        an UnmappedSourceConceptError.
        """
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

    def try_map_atom[A: _Attribute[dc.ConceptId]](
        self, identifier: dc.ConceptCodeVocab, expected_class: type[A]
    ) -> list[A]:
        """
        Try to map an atomic attribute by its source identifier and check if
        it's of the expected class. If the attribute is not of the expected
        class, raises an UnmappedSourceConceptError.
        """
        try:
            ids = self._concept_map[identifier]
        except KeyError as e:
            self.logger.error(
                f"Tried mapping {expected_class.__name__} "
                f"by {identifier} which is not a valid attribute."
            )
            raise UnmappedSourceConceptError(
                f"Invalid attribute: {identifier} for {expected_class.__name__}"
            ) from e

        return [
            self._try_get_atom(dc.ConceptId(id), expected_class) for id in ids
        ]

    def _get_unbound_strength_variations[S: dc.Strength](
        self,
        strength_data: dc.ForeignStrength,
        expected_class: type[S],
    ) -> Generator[S, None, None]:
        """
        Translate a slice of DS_STAGE dosage information into
        RxNorm-native strength data.

        As units may be mapped with precedence, this method may yield multiple
        strength data variations for a single input.
        """
        # NOTE: Pyright complains about the non-covariant generator return type,
        # but it is.

        match expected_class:
            case dc.SolidStrength:
                assert strength_data.amount_value is not None
                assert strength_data.amount_unit is not None
                for scaled_v, true_unit in self._translate_strength_measure(
                    strength_data.amount_value, strength_data.amount_unit
                ):
                    yield dc.SolidStrength(scaled_v, true_unit)  # pyright: ignore[reportReturnType] # noqa: E501
                return

            case dc.LiquidConcentration:
                assert strength_data.numerator_value
                assert strength_data.numerator_unit
                assert strength_data.denominator_unit
                num_permut = self._translate_strength_measure(
                    strength_data.numerator_value, strength_data.numerator_unit
                )
                den_permut = self._translate_strength_measure(
                    PyRealNumber(1), strength_data.denominator_unit
                )
                for (scaled_n, n_unit), (scaled_d, d_unit) in product(
                    num_permut, den_permut
                ):
                    yield dc.LiquidConcentration(  # pyright: ignore[reportReturnType] # noqa: E501
                        numerator_value=scaled_n / scaled_d,
                        numerator_unit=n_unit,
                        denominator_unit=d_unit,
                    )
                return

            case dc.LiquidQuantity:
                assert strength_data.numerator_value
                assert strength_data.numerator_unit
                assert strength_data.denominator_value
                assert strength_data.denominator_unit
                num_permut = self._translate_strength_measure(
                    strength_data.numerator_value, strength_data.numerator_unit
                )
                den_permut = self._translate_strength_measure(
                    strength_data.denominator_value,
                    strength_data.denominator_unit,
                )
                for (scaled_n, n_unit), (scaled_d, d_unit) in product(
                    num_permut, den_permut
                ):
                    yield dc.LiquidQuantity(  # pyright: ignore[reportReturnType] # noqa: E501
                        numerator_value=scaled_n,
                        numerator_unit=n_unit,
                        denominator_value=scaled_d,
                        denominator_unit=d_unit,
                    )
                return

            case dc.GasPercentage:
                # Those are static
                assert strength_data.numerator_value
                assert strength_data.numerator_unit
                pct_map = self._unit_map[strength_data.numerator_unit]
                for pct_u, conv in pct_map.items():  # What if?...
                    yield dc.GasPercentage(  # pyright: ignore[reportReturnType] # noqa: E501
                        conv * strength_data.numerator_value, pct_u
                    )
                return
            case _:
                # Unreachable
                raise ValueError(
                    f"Unknown strength configuration: {expected_class}"
                )

    def _get_ingredient_permutations(
        self, ingredients: Sequence[dc.Ingredient[dc.ConceptCodeVocab]]
    ) -> Generator[_PrecedentedStrengthData[None], None, None]:
        """
        Get all possible permutations of the node that has only ingredients and
        no strength data.
        """

        # Stand-in for the strength data
        _strength = None

        precedented_targets: list[
            tuple[int, dc.BoundStrength[dc.ConceptId, None]]
        ]
        all_ingredient_permutations: list[
            list[tuple[int, dc.BoundStrength[dc.ConceptId, None]]]
        ] = []
        for ingredient in ingredients:
            targets: list[dc.Ingredient[dc.ConceptId]] = self.try_map_atom(
                ingredient.identifier, dc.Ingredient
            )
            precedented_targets = [
                (i, (t, _strength)) for i, t in enumerate(targets)
            ]
            all_ingredient_permutations.append(precedented_targets)

        for combo_permutation in product(*all_ingredient_permutations):
            precedence = sum(p for p, _ in combo_permutation)
            strengths = SortedTuple(i for _, i in combo_permutation)
            yield precedence, strengths

    def get_attribute_permutations(
        self,
        dose_form: dc.DoseForm[dc.ConceptCodeVocab] | None,
        brand_name: dc.BrandName[dc.ConceptCodeVocab] | None,
        supplier: dc.Supplier[dc.ConceptCodeVocab] | None,
    ) -> AttributePermutations:
        """
        Get all possible permutations of the node's atomic attributes in order
        of precedence.
        """

        # NOTE: Pyright ignore comments are here because types are checked
        # explicitly on runtime inside try_map_atom and _try_get_atom methods
        all_attrs = []
        for attr, class_ in (
            (dose_form, dc.DoseForm),
            (brand_name, dc.BrandName),
            (supplier, dc.Supplier),
        ):
            if attr is not None:
                attrs = self.try_map_atom(attr.identifier, class_)  # pyright: ignore[reportUnknownVariableType]  # noqa: E501
            else:
                attrs = [None]
            all_attrs.append(list(enumerate(attrs)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # noqa: E501

        return AttributePermutations(*all_attrs)

    def _translate_strength_row[S: dc.Strength](
        self,
        ingredient: dc.Ingredient[dc.ConceptCodeVocab],
        strength: dc.ForeignStrength,
        expected_class: type[S],
    ) -> Generator[_PrecedentedBoundStrength[S], None, None]:
        """
        Translate a single strength row into a sequence of possible strength
        data variations in RxNorm-native units, including the ingredient and
        its precedence.

        """
        ingredient_targets: list[dc.Ingredient[dc.ConceptId]]
        ingredient_targets = self.try_map_atom(
            ingredient.identifier, dc.Ingredient
        )

        strengths = self._get_unbound_strength_variations(
            strength, expected_class
        )

        for (i, ing_id), s in product(enumerate(ingredient_targets), strengths):
            yield i, (ing_id, s)

    def _get_strength_permutations[S: dc.Strength](
        self,
        rows: Sequence[dc.BoundForeignStrength],
        expected_class: type[S],
    ) -> Generator[_PrecedentedStrengthData[S], None, None]:
        """
        Get all possible permutations of the node's strength data in order of
        precedence of the contributing ingredient mappings.

        Precedence returned is accumulated as sum of all precedences of the
        contributing ingredients.
        """
        row_permutations: list[list[_PrecedentedBoundStrength[S]]] = []
        for ingredient, strength in rows:
            if strength is None:
                raise ValueError(
                    "Strength data is None. Such cases must be passed to "
                    "_get_ingredient_only_permutations method."
                )

            permutations = self._translate_strength_row(
                ingredient, strength, expected_class
            )
            row_permutations.append(list(permutations))

        for strength_permutation in product(*row_permutations):
            precedence = sum(p for p, _ in strength_permutation)
            strength = SortedTuple(s for _, s in strength_permutation)
            yield precedence, strength

    def _get_ingredient_only_permutations(
        self,
        ingredients: Sequence[dc.Ingredient[dc.ConceptCodeVocab]],
    ) -> Generator[_PrecedentedStrengthData[None], None, None]:
        """
        Get all possible permutations of the node that has only ingredients and
        no strength data.
        """
        row_permutations: list[list[_PrecedentedIngredient]] = []
        mappings: list[dc.Ingredient[dc.ConceptId]]
        for ingredient in ingredients:
            mappings = self.try_map_atom(ingredient.identifier, dc.Ingredient)
            row_permutations.append(list(enumerate(mappings)))

        for ing_permutation in product(*row_permutations):
            precedence = sum(p for p, _ in ing_permutation)
            ings_combo = SortedTuple((i, None) for _, i in ing_permutation)
            yield precedence, ings_combo
