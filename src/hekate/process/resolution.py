"""
Contains Resolver class, which is used to disambiguate between the available
mapping results of virtual nodes to RxNorm/RxNorm-Extension concepts, and to
choose a single virtual node to represent the concept.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence  # for type annotations
from abc import abstractmethod, ABC  # for resolver protocol
from itertools import chain, groupby, product  # for concise iterations
from typing import NamedTuple, override  # for ResultCharacteristics

import polars as pl  # for metadata table manipulation
from csv_read import athena  # for concept metadata
from rx_model import drug_classes as dc  # For generic interfaces and CDC
from utils import exceptions
from utils.classes import (
    PyRealNumber,  # For type agnostic numeric values
    SortedTuple,  # Constructing strength combinations from multiple CDCs
)

type _VirtualDrugNode = dc.ForeignDrugNode[dc.Strength | None]
type _Terminal = dc.HierarchyNode[dc.ConceptId]

type _CDC = dc.ClinicalDrugComponent[dc.ConceptId, dc.UnquantifiedStrength]
type _StrengthData = SortedTuple[
    dc.BoundStrength[dc.ConceptId, dc.Strength | None]
]

# Type-agnostic
_ZERO = PyRealNumber(0)


# TODO: Document the order elsewhere
# TODO: Include Precise Ingreidient genericity degree
class ResultCharacteristics(NamedTuple):
    """
    Tuple of numeric grades that quantify the degree of similarity between
    the ForeignDrugNode and the target HierarchyNode.

    The order of disambiguation is as follows:
    1. Lowest precedence Ingredient match. Drugs with multiple
    ingredients can simply sum the precedence difference for this
    matter – precedence collision between multi-ingredient drugs
    is an extremely rare edge case, not warranting a more complex
    logic. Precedence for packs is also summed.
    2. Lowest relative difference with strength denominator value, if
    present. This is usually not something that depends on ingredient
    salt, and usually is just the integer count of milliliters. Ignored for
    packs, as precedence logic is unclear.
    3. Lowest precedence Dose Form match.
    4. Lowest precedence Brand Name match.
    5. Lowest precedence Supplier match.
    6. Now that disambiguation on attribute precedence is impossible,
     closest match on Strength values, determined for example by
     average of relative component difference.
    7. RxNorm over RxNorm Extension target.
    8. Lower valid_start_date of target.
    9. Lower concept_id as a final tie-breaker.
    """

    # NOTE: The order of the fields is important.
    ingredient_diff: int
    denominator_diff: PyRealNumber
    dose_form_diff: int
    brand_name_diff: int
    supplier_diff: int
    strength_diff: PyRealNumber
    # is_extension: bool  # 0 for RxNorm, 1 for RxNorm Extension
    is_extension: int  # May be above 1 for multicomponent mapping
    valid_start_date: int  # int in YYYYMMDD format
    concept_id: dc.ConceptId  # int

    class StrengthDiff(NamedTuple):
        """
        Tuple of numeric grades that quantify the degree of similarity between
        the Strength data of the ForeignDrugNode and the target DrugNode.
        """

        numerator_score: PyRealNumber
        denominator_score: PyRealNumber

        @classmethod
        def from_strength_pair(
            cls,
            source_strength: _StrengthData,
            target_strength: _StrengthData,
        ) -> ResultCharacteristics.StrengthDiff:
            n_diff: PyRealNumber
            d_diff: PyRealNumber
            t_str = target_strength[0][1]
            s_str = source_strength[0][1]
            if isinstance(t_str, dc.LiquidQuantity):
                # LiquidQuantity is the most specific strength type, so
                # source can only be LiquidQuantity
                assert isinstance(s_str, dc.LiquidQuantity)
                d_diff = abs(t_str.denominator_value - s_str.denominator_value)
            elif t_str is not None:
                # Always 0 for unquantified strengths
                d_diff = _ZERO
            else:
                # No strength data, noop
                return cls(
                    numerator_score=_ZERO,
                    denominator_score=_ZERO,
                )

            # Accumulate relative differences
            n_diff = _ZERO
            for (_, s_str), (_, t_str) in zip(source_strength, target_strength):
                if isinstance(t_str, dc.SolidStrength):
                    # Only possible source counterpart
                    assert isinstance(s_str, dc.SolidStrength)
                    assert isinstance(t_str, dc.SolidStrength)
                    if s_str.amount_value == _ZERO:
                        # Avoid division by zero for inert ingredients
                        continue
                    t_val = t_str.amount_value
                    s_val = s_str.amount_value
                else:
                    # Guaranteed to have a numerator
                    # NOTE: we operate with unquantified strength to avoid
                    # duplicate code
                    assert s_str and t_str
                    norm_s_str = s_str.get_unquantified()
                    norm_t_str = t_str.get_unquantified()
                    assert isinstance(
                        norm_s_str, (dc.GasPercentage, dc.LiquidConcentration)
                    )
                    assert isinstance(
                        norm_t_str, (dc.GasPercentage, dc.LiquidConcentration)
                    )
                    t_val = norm_t_str.numerator_value
                    s_val = norm_s_str.numerator_value

                n_diff += abs(t_val - s_val) / s_val

            return cls(numerator_score=n_diff, denominator_score=d_diff)

    @classmethod
    def from_drug_pair(
        cls,
        source: _VirtualDrugNode,
        target: _Terminal,
        concept_metadata: pl.DataFrame,
    ) -> ResultCharacteristics:
        """
        Calculate the numeric grades for the similarity between the source
        and target nodes.
        """
        try:
            metadata = concept_metadata.filter(
                pl.col("concept_id") == target.identifier
            )
        except pl.exceptions.ColumnNotFoundError as e:
            raise exceptions.ResolutionError(
                "Unexpected metadata structure"
            ) from e

        if len(metadata) != 1:
            raise exceptions.ResolutionError(
                f"Unique concept ID {target.identifier} not found in the "
                f"metadata."
            )

        if isinstance(target, dc.MarketedProductNode):
            # Unpack
            target = target.terminal_parent

        match target:
            case dc.Ingredient():
                # No strength data
                return cls(
                    source.precedence_data.ingredient_diff,
                    _ZERO,
                    0,
                    0,
                    0,
                    _ZERO,
                    is_extension=metadata[0, "vocabulary_id"]
                    == "RxNorm Extension",
                    valid_start_date=metadata[0, "valid_start_date"],
                    concept_id=target.identifier,
                )
            case dc.DrugNode():
                n_diff, d_diff = cls.StrengthDiff.from_strength_pair(
                    source_strength=source.get_strength_data(),
                    target_strength=target.get_strength_data(),  # pyright: ignore[reportUnknownArgumentType]  # noqa: E501
                )
                return cls(
                    ingredient_diff=source.precedence_data.ingredient_diff,
                    denominator_diff=d_diff,
                    dose_form_diff=source.precedence_data.dose_form_diff,
                    brand_name_diff=source.precedence_data.brand_name_diff,
                    supplier_diff=source.precedence_data.supplier_diff,
                    strength_diff=n_diff,
                    is_extension=metadata[0, "vocabulary_id"]
                    == "RxNorm Extension",
                    valid_start_date=metadata[0, "valid_start_date"],
                    concept_id=target.identifier,
                )

            case dc.PackNode():
                # Programming error: PackNode should never become a target for
                # a ForeignDrugNode
                raise ValueError(
                    f"PackNode {target.identifier} should never be a target "
                    f"for a ForeignDrugNode {source.identifier}."
                )
            case _:
                raise NotImplementedError(
                    f"Unknown target type {target.__class__.__name__}."
                )

    @classmethod
    def from_component_combo(
        cls,
        source: _VirtualDrugNode,
        targets: Sequence[_CDC],
        concept_metadata: pl.DataFrame,
    ) -> ResultCharacteristics:
        """
        Calculate the numeric grades for the similarity between the source
        node and a list of components that each correspond to one of its
        strength entries.

        This is intended for deduplication of Clinical Drug Component targets
        grouped by the same ingredient.
        """
        try:
            metadata = concept_metadata.filter(
                pl.col("concept_id").is_in([t.identifier for t in targets])
            )
        except pl.exceptions.ColumnNotFoundError as e:
            raise exceptions.ResolutionError(
                "Unexpected metadata structure"
            ) from e

        if len(metadata) != len(targets):
            raise exceptions.ResolutionError(
                f"Not all concept IDs of {[t.identifier for t in targets]} "
                f"found in the metadata."
            )

        n_diff, d_diff = cls.StrengthDiff.from_strength_pair(
            source_strength=source.get_strength_data(),
            target_strength=SortedTuple(
                target.get_strength_data()[0] for target in targets
            ),
        )

        # Use highest valid_start_date and concept_id for grading the combo
        max_start_date = metadata["valid_start_date"].max()
        max_concept_id = metadata["concept_id"].max()
        assert isinstance(max_start_date, int)
        assert isinstance(max_concept_id, int)

        return cls(
            ingredient_diff=source.precedence_data.ingredient_diff,
            denominator_diff=abs(d_diff),
            dose_form_diff=0,  # Not in CDC
            brand_name_diff=0,  # Not in CDC
            supplier_diff=0,  # Not in CDC
            strength_diff=n_diff,
            is_extension=sum(metadata["vocabulary_id"] == "RxNorm Extension"),
            valid_start_date=max_start_date,
            concept_id=dc.ConceptId(max_concept_id),
        )

    @classmethod
    def from_pack_pair(
        cls,
        source: dc.ForeignPackNode,
        target: dc.PackNode[dc.ConceptId],
        concept_metadata: pl.DataFrame,
    ) -> ResultCharacteristics:
        """
        Calculate the numeric grades for the similarity between the source
        and target nodes.
        """
        try:
            metadata = concept_metadata.filter(
                pl.col("concept_id") == target.identifier
            )
        except pl.exceptions.ColumnNotFoundError as e:
            raise exceptions.ResolutionError(
                "Unexpected metadata structure"
            ) from e

        if len(metadata) != 1:
            raise exceptions.ResolutionError(
                f"Unique concept ID {target.identifier} not found in the "
                f"metadata."
            )

        return cls(
            ingredient_diff=source.precedence_data.ingredient_diff,
            denominator_diff=_ZERO,  # No strength data in packs
            dose_form_diff=source.precedence_data.dose_form_diff,
            brand_name_diff=source.precedence_data.brand_name_diff,
            supplier_diff=source.precedence_data.supplier_diff,
            strength_diff=_ZERO,  # No strength data in packs
            is_extension=metadata[0, "vocabulary_id"] == "RxNorm Extension",
            valid_start_date=metadata[0, "valid_start_date"],
            concept_id=target.identifier,
        )


class _Resolver[
    PrototypeClass: dc.ForeignNodePrototype | dc.ForeignPackNodePrototype,
    ForeignClass: dc.ForeignDrugNode[dc.Strength | None] | dc.ForeignPackNode,
](ABC):
    """
    Resolver class is used to disambiguate between the available mapping results
    of virtual nodes to RxNorm/RxNorm-Extension concepts, and to choose a single
    virtual node to represent the concept.

    Contains implementation of disambiguation of source node representation
    (combination) translation, with approximation of Build-RxE expectation.
    """

    def __init__(
        self,
        source_prototype: PrototypeClass,
        mapping_results: Mapping[ForeignClass, Sequence[_Terminal]],
        logger: logging.Logger,
        concept_handle: athena.ConceptTable,
    ):
        self.logger: logging.Logger = logger.getChild(
            self.__class__.__name__
        ).getChild(str(source_prototype.identifier))

        # Operands
        self.source_prototype: PrototypeClass = source_prototype
        self.mapping_results: Mapping[ForeignClass, Sequence[_Terminal]] = (
            mapping_results
        )

        # Concept data lookup
        self.concept_handle: athena.ConceptTable = concept_handle

    @abstractmethod
    def pick_omop_mapping(
        self,
    ) -> Sequence[dc.HierarchyNode[dc.ConceptId]]:
        """
        Pick the mapping result that has the highest priority and is ready for
        inclusion.

        Will discard mapping options according to conventional class preference
        order.
        """


class DrugResolver(
    _Resolver[
        dc.ForeignNodePrototype,
        dc.ForeignDrugNode[dc.Strength | None],
    ]
):
    @override
    def pick_omop_mapping(self) -> list[dc.HierarchyNode[dc.ConceptId]]:
        """
        Pick the mapping result that has the highest priority and is ready for
        inclusion.

        Will discard mapping options according to conventional class preference
        order.

        May return multiple options if the best options belong to Ingredient or
        Clinical Drug Component class.
        """
        # First, keep only the nodes that contain best result by class
        # preference order.
        best_class = type(
            min(
                chain(*self.mapping_results.values()),
                key=lambda node: dc.DRUG_CLASS_PREFERENCE_ORDER.index(
                    # TODO: use CCId enum
                    type(node)  # pyright: ignore[reportArgumentType]
                ),
            )
        )

        if best_class == dc.Ingredient:
            # Sort nodes by ingredient precedence and return the best one
            # NOTE: It is expected to be mapped to one or more ingredients,
            # number matching the source node.
            best_node = min(
                self.mapping_results,
                key=lambda foreign: foreign.precedence_data.ingredient_diff,
            )
            return list(self.mapping_results[best_node])

        elif best_class == dc.ClinicalDrugComponent:
            # 1. Sort nodes by ingredient precedence
            nodes_by_ingredient = sorted(
                self.mapping_results,
                key=lambda foreign: foreign.precedence_data.ingredient_diff,
            )

            # 2. Find nodes with no terminal ingredients (meaning all components
            # exist)
            cdc_nodes: list[_VirtualDrugNode] = []
            for node in nodes_by_ingredient:
                if not any(
                    isinstance(terminal, dc.Ingredient)
                    for terminal in self.mapping_results[node]
                ):
                    cdc_nodes.append(node)

            # 2.a. No nodes with all components found, return ingredients of
            # the best node
            if not cdc_nodes:
                # Only monoingredients here, ignore strength
                return list(self.mapping_results[nodes_by_ingredient[0]])
            # 2.b. Rate and deduplicate the components grouped by ingredient
            else:
                scored: Sequence[
                    tuple[ResultCharacteristics, tuple[_CDC, ...]]
                ] = []
                for node in cdc_nodes:
                    components: Sequence[_CDC]
                    components = self.mapping_results[node]  # pyright: ignore[reportAssignmentType]  # noqa: E501
                    by_ingredient = groupby(
                        sorted(components, key=lambda cdc: cdc.ingredient),
                        key=lambda cdc: cdc.ingredient,
                    )
                    component_groups = [list(g) for _, g in by_ingredient]
                    metadata = self.concept_handle.get_metadata([
                        cdc.identifier for cdc in components
                    ])
                    scored.extend(
                        (
                            ResultCharacteristics.from_component_combo(
                                node, combo, metadata
                            ),
                            combo,
                        )
                        for combo in product(*component_groups)
                    )

                # Returns best scored combination of components
                return list(min(scored)[1])
        else:  # All multi-ingredient classes
            all_pairs: list[tuple[_VirtualDrugNode, _Terminal]] = list(
                (source, target)
                for source, targets in self.mapping_results.items()
                for target in targets
                if isinstance(target, best_class)
            )
            metadata = self.concept_handle.get_metadata([
                target.identifier for _, target in all_pairs
            ])
            best_node: _Terminal = min(
                all_pairs,
                key=lambda pair: ResultCharacteristics.from_drug_pair(
                    *pair, metadata
                ),
            )[1]

            return [best_node]


class PackResolver(
    _Resolver[
        dc.ForeignPackNodePrototype,
        dc.ForeignPackNode,
    ]
):
    @override
    def pick_omop_mapping(self) -> Sequence[dc.HierarchyNode[dc.ConceptId]]:
        """
        Pick the mapping result that has the highest priority and is ready for
        inclusion.

        Will discard mapping options according to conventional class preference
        order.

        If no mapping options are given, will return component drug targets.
        """

        # Find nodes which have at least one pack result
        nodes_with_pack: dict[dc.ForeignPackNode, Sequence[_Terminal]] = {
            node: list(filter(lambda t: isinstance(t, dc.PackNode), targets))
            for node, targets in self.mapping_results.items()
            if any(isinstance(t, dc.PackNode) for t in targets)
        }

        if not nodes_with_pack:
            # No pack nodes found, return component drug targets
            best_case_node: dc.ForeignPackNode = min(
                self.mapping_results,
                key=lambda fpn: (
                    # 1. Lowest precedence of cumulative attribute precedence
                    fpn.precedence_data,
                    # 2. Lowest value of pack entry identifier (concept_id)
                    fpn.entries[0].drug.identifier,  # already in SortedTuple
                ),
            )

            return [entry.drug for entry in best_case_node.entries]

        # Keep only the nodes that contain best result by class
        best_class = type(
            min(
                chain(*nodes_with_pack.values()),
                key=lambda node: dc.PACK_CLASS_PREFERENCE_ORDER.index(
                    # TODO: use CCId enum
                    type(node)  # pyright: ignore[reportArgumentType]
                ),
            )
        )
        assert best_class in dc.PACK_CLASS_PREFERENCE_ORDER

        all_pairs: Sequence[
            tuple[dc.ForeignPackNode, dc.PackNode[dc.ConceptId]]
        ] = list(
            (source, target)
            for source, targets in nodes_with_pack.items()
            for target in targets
            if isinstance(target, best_class)
        )
        metadata = self.concept_handle.get_metadata([
            target.identifier for _, target in all_pairs
        ])
        best_node: dc.PackNode[dc.ConceptId] = min(
            all_pairs,
            key=lambda pair: ResultCharacteristics.from_pack_pair(
                *pair, metadata
            ),
        )[1]

        return [best_node]
