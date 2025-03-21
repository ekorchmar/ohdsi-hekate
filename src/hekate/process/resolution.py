"""
Contains Resolver class, which is used to disambiguate between the available
mapping results of virtual nodes to RxNorm/RxNorm-Extension concepts, and to
choose a single virtual node to represent the concept.
"""

import logging
from collections.abc import Mapping
from itertools import chain
from typing import NamedTuple

import polars as pl
from csv_read import athena
from rx_model import drug_classes as dc
from utils import exceptions

type _VirtualNode = dc.ForeignDrugNode[dc.Strength | None]
type _Terminal = dc.DrugNode[dc.ConceptId, dc.Strength | None]
type _TerminalClass = type[_Terminal]


class ResultCharacteristics(NamedTuple):
    # TODO: Document the order elsewhere
    """
    Tuple of numeric grades that quantify the degree of similarity between
    the ForeignDrugNode and the target DrugNode.

    The order of disambiguation is as follows:
    1. Lowest precedence Ingredient match. Drugs with multiple
    ingredients can simply sum the precedence difference for this
    matter – precedence collision between multi-ingredient drugs
    is an extremely rare edge case, not warranting a more complex
    logic.
    2. Lowest relative difference with strength denominator value, if
    present. This is usually not something that depends on ingredient
    salt, and usually is just the integer count of milliliters.
    3. Lowest precedence Dose Form match.
    4. Lowest precedence Brand Name match.
    5. Lowest precedence Supplier match.
    6. Now that disambiguation on attribute precedence is impossible,
     closest match on Strength values, determined for example by
     average of relative component difference.
    7. RxNorm over RxNorm Extension target.
    8. Lower valid_start_date of target.
    9. Lower concept_id if we're desperate.
    """

    # NOTE: The order of the fields is important.
    ingredient_diff: int
    denominator_diff: float
    dose_form_diff: int
    brand_name_diff: int
    supplier_diff: int
    strength_diff: float
    is_extension: bool  # 0 for RxNorm, 1 for RxNorm Extension
    valid_start_date: int  # int in YYYYMMDD format
    concept_id: dc.ConceptId  # int

    @classmethod
    def from_pair(
        cls,
        source: _VirtualNode,
        target: _Terminal,
        concept_metadata: pl.DataFrame,
    ) -> "ResultCharacteristics":
        """
        Calculate the numeric grades for the similarity between the source
        and target nodes.
        """
        try:
            metadata = concept_metadata.filter(
                pl.col("concept_id") == target.identifier
            ).select(["vocabulary_id", "valid_start_date"])
        except pl.exceptions.ColumnNotFoundError as e:
            raise exceptions.ResolutionError(
                f"Concept ID {target.identifier} not found in the metadata."
            ) from e

        if len(metadata) != 1:
            raise exceptions.ResolutionError(
                f"Unique concept ID {target.identifier} not found in the "
                f"metadata."
            )

        n_diff: float
        d_diff: float
        _, t_str = next(iter(target.get_strength_data()))
        _, s_str = next(iter(source.strength_data))
        if isinstance(t_str, dc.LiquidQuantity):
            # LiquidQuantity is the most specific strength type, so:
            assert isinstance(s_str, dc.LiquidQuantity)
            d_diff = float(t_str.denominator_value - s_str.denominator_value)
            n_diff = float(t_str.numerator_value - s_str.numerator_value)
        else:
            d_diff = 0.0
            if isinstance(t_str, dc.SolidStrength):
                # Only possible source counterpart
                assert isinstance(s_str, dc.SolidStrength)
                n_diff = float(t_str.amount_value - s_str.amount_value)
            elif t_str is not None:
                # Gas or Liquid Concentration
                assert isinstance(s_str, type(t_str))
                n_diff = float(t_str.numerator_value - s_str.numerator_value)
            else:
                n_diff = 0.0

        return cls(
            ingredient_diff=source.precedence_data.ingredient_diff,
            denominator_diff=abs(d_diff),
            dose_form_diff=source.precedence_data.dose_form_diff,
            brand_name_diff=source.precedence_data.brand_name_diff,
            supplier_diff=source.precedence_data.supplier_diff,
            strength_diff=abs(n_diff),
            is_extension=metadata[0, "vocabulary_id"] == "RxNorm Extension",
            valid_start_date=metadata[0, "valid_start_date"],
            concept_id=target.identifier,
        )


class Resolver:
    """
    Resolver class is used to disambiguate between the available mapping results
    of virtual nodes to RxNorm/RxNorm-Extension concepts, and to choose a single
    virtual node to represent the concept.

    Contains implementation of disambiguation of source node representation
    (combination) translation, with approximation of Build-RxE expectation.
    """

    def __init__(
        self,
        source_definition: dc.ForeignNodePrototype,
        mapping_results: Mapping[
            dc.ForeignDrugNode[dc.Strength | None], list[_Terminal]
        ],
        logger: logging.Logger,
        concept_handle: athena.ConceptTable,
    ):
        self.logger: logging.Logger = logger.getChild(self.__class__.__name__)

        # Operands
        self.source_definition: dc.ForeignNodePrototype = source_definition
        self.mapping_results: Mapping[
            dc.ForeignDrugNode[dc.Strength | None], list[_Terminal]
        ] = mapping_results

        # Concept data lookup
        self.concept_handle: athena.ConceptTable = concept_handle

    def disambiguate_targets(self) -> None:
        """
        Disambiguate the mapping results to a subset of non-overlapping
        concepts.
        """
        raise NotImplementedError

    def resolve(self) -> dc.DrugNode[dc.ConceptId, dc.Strength | None]:
        """
        Resolve the mapping results to a single virtual node.

        Chooses a node with both deepest hierarchy penetration and highest
        priority by attribute mapping precedence, numeric similarity and, if
        still necessary, by metadata.
        """
        raise NotImplementedError

    def pick_omop_mapping(
        self,
    ) -> list[dc.DrugNode[dc.ConceptId, dc.Strength | None]]:
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
        best_class = min(
            chain(*self.mapping_results.values()),
            key=lambda node: dc.DRUG_CLASS_PREFERENCE_ORDER.index(
                node.__class__
            ),
        ).__class__

        nodes_having_best_class = {
            n: list(filter(lambda node: isinstance(node, best_class), t))
            for n, t in self.mapping_results.items()
        }

        for k in [k for k, v in nodes_having_best_class.items() if not v]:
            del nodes_having_best_class[k]

        del nodes_having_best_class
        raise NotImplementedError
