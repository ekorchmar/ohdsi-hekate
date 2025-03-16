"""
Contains Resolver class, which is used to disambiguate between the available
mapping results of virtual nodes to RxNorm/RxNorm-Extension concepts, and to
choose a single virtual node to represent the concept.
"""

from collections.abc import Mapping
import logging
from rx_model import drug_classes as dc


class Resolver:
    """
    Resolver class is used to disambiguate between the available mapping results
    of virtual nodes to RxNorm/RxNorm-Extension concepts, and to choose a single
    virtual node to represent the concept.
    """

    def __init__(
        self,
        source_definition: dc.ForeignNodePrototype,
        mapping_results: Mapping[
            dc.ForeignDrugNode[dc.Strength | None],
            dc.DrugNode[dc.ConceptId, dc.Strength | None],
        ],
        logger: logging.Logger,
    ):
        self.logger: logging.Logger = logger.getChild(self.__class__.__name__)

        # Operands
        self.source_definition: dc.ForeignNodePrototype = source_definition
        self.mapping_results: Mapping[
            dc.ForeignDrugNode[dc.Strength | None],
            dc.DrugNode[dc.ConceptId, dc.Strength | None],
        ] = mapping_results

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
        raise NotImplementedError
