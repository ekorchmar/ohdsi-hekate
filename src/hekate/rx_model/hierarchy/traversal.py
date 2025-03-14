"""
Hosts a visitor class that traverses the RxHierarchy tree (BFS) and finds a new
appropriate location for a new node.

Note: PruneSearch is raised to prune the search tree while traversing a graph.
It will not stop the search, only discard the successor nodes.
"""

from typing import override
import logging

import rustworkx as rx
from rx_model.drug_classes import (
    ALLOWED_DRUG_MULTIMAP,
    DRUG_CLASS_PREFERENCE_ORDER,
    ConceptId,
    DrugNode,
    ForeignDrugNode,
    Ingredient,
    Strength,
)
from rx_model.hierarchy.hosts import NodeIndex, RxHierarchy
from utils.utils import get_first_dict_value

# TODO: this should eventually become a continuous value to track preference
# across precedence levels.
type NodeAcceptance = bool


class DrugNodeFinder(rx.visit.BFSVisitor):
    """
    A visitor class that traverses the RxHierarchy tree (BFS) and finds a new
    appropriate location for a new node.
    """

    def __init__(
        self,
        node: ForeignDrugNode[Strength | None],
        hierarchy: RxHierarchy[ConceptId],
        logger: logging.Logger,
    ):
        """
        Initializes the visitor with the hierarchy and the new node to be
        tested.
        """
        self.node: ForeignDrugNode[Strength | None] = node
        self.logger: logging.Logger = logger.getChild(
            f"{self.__class__.__name__}({node.identifier})"
        )

        self.history: dict[NodeIndex, NodeAcceptance] = {}

        self.aim_for: type[DrugNode[ConceptId, Strength | None]] = (
            self.node.best_case_class()
        )

        # Indicates how many targets of ALLOWED_DRUG_MULTIMAP should
        # the node end up with.
        self.number_components: int

        # Performance optimization: once enough MULTIMAP entries are found, the
        # rest of the Ingredient/CDC nodes can be pruned.
        self.__matched_ingredient_count: int = 0

        if self.node.is_multi():
            self.number_components = len(self.node.strength_data)
        else:
            self.number_components = 1

        self.hierarchy: RxHierarchy[ConceptId] = hierarchy

        # Container for the results
        self.final_nodes: dict[
            NodeIndex, DrugNode[ConceptId, Strength | None]
        ] = {}

    @override
    def discover_vertex(self, v: NodeIndex) -> None:
        """
        Called when a new vertex is discovered.
        """
        drug_node = self.hierarchy.graph[v]

        if isinstance(drug_node, Ingredient):
            if self.__matched_ingredient_count >= self.number_components:
                # Stop taking any new ingredient branches
                self._remember_node(v, False)
                raise rx.visit.PruneSearch
        else:
            # Check if the node's predecessors are available in history
            if any(
                p_idx not in self.final_nodes
                for p_idx in self.hierarchy.graph.predecessor_indices(v)
            ):
                # Not all predecessors are accepted
                self._remember_node(v, False)
                raise rx.visit.PruneSearch

        acceptance: NodeAcceptance = drug_node.is_superclass_of(self.node)
        self._remember_node(v, acceptance)

        if not acceptance:
            # None of the descendants will match
            raise rx.visit.PruneSearch
        else:
            self._accept_node(v)

    def _accept_node(self, v: NodeIndex) -> None:
        """
        Accepts a node and remembers it.
        """
        drug_node = self.hierarchy.graph[v]
        self.final_nodes[v] = drug_node
        if isinstance(drug_node, self.aim_for):
            # We are almost there. However, this might be a node for
            # multi-mapping.
            can_be_multi = self.number_components > 1 and isinstance(
                drug_node, ALLOWED_DRUG_MULTIMAP
            )

            if not can_be_multi:
                # This is the node we are looking for. But we still may
                # match sister nodes.
                raise rx.visit.PruneSearch

    def _remember_node(self, v: NodeIndex, acceptance: NodeAcceptance) -> None:
        """
        Remembers the acceptance of a node.
        """
        self.history[v] = acceptance

    def start_search(self) -> None:
        """
        Starts the depth-first search on the hierarchy, starting from the
        ingredient roots.
        """

        self.logger.debug("Starting search for the matching node")
        self.__matched_ingredient_count = 0

        # We always know the first match level
        ing_indices = [
            self.hierarchy.ingredients[ing]
            for ing, _ in self.node.get_strength_data()
        ]

        rx.bfs_search(self.hierarchy.graph, ing_indices, self)

    def get_search_results(
        self,
    ) -> dict[NodeIndex, DrugNode[ConceptId, Strength | None]]:
        """
        Returns the search results.
        """
        with_dups = self._get_raw_search_results()

        # Base case
        if len(with_dups) == 1:
            return with_dups

        # If there are multple options, check if the node is multi-mappable
        if len(with_dups) == self.number_components:
            if isinstance(list(with_dups.values())[0], ALLOWED_DRUG_MULTIMAP):
                # NOTE: Check if there are repeated ingredients
                ings = set()
                for node in with_dups.values():
                    ings.update(ing for ing, _ in node.get_strength_data())
                if len(ings) == self.number_components:
                    return with_dups

        # Disambiguate the results
        return self._disambiguate_search_results(with_dups)

    def _get_raw_search_results(
        self,
    ) -> dict[NodeIndex, DrugNode[ConceptId, Strength | None]]:
        """
        Returns NOT DISAMBIGUATED best case nodes found during the search.
        """
        if not self.final_nodes:
            raise ValueError("No nodes found. Did you call start_search()?")

        # The best node is guaranteed to be the last one in BFS
        reversed_nodes = list(self.final_nodes.keys())[::-1]
        best_node = self.final_nodes[reversed_nodes[0]]

        best_class_: type[DrugNode[Strength | None]]
        for best_class_ in DRUG_CLASS_PREFERENCE_ORDER:
            if isinstance(best_node, best_class_):
                break
        else:
            raise ValueError(
                f"Encountered unexpected type {best_node.__class__.__name__} "
                f"which node {best_node.identifier} is an instance of."
            )

        best_nodes = {reversed_nodes[0]: best_node}
        for idx in best_nodes:
            node = best_nodes[idx]
            if isinstance(node, best_class_):
                best_nodes[idx] = node

        self.logger.debug(
            f"Found {len(best_nodes)} best case nodes: {best_nodes}"
        )
        return best_nodes

    def _disambiguate_search_results(
        self, choices: dict[NodeIndex, DrugNode[ConceptId, Strength | None]]
    ) -> dict[NodeIndex, DrugNode[ConceptId, Strength | None]]:
        """
        Disambiguates the search results.
        """
        # NOTE: Algorithm for disambiguation is a subject of an active
        # discussion between the members of the OHDSI working group.
        # This is a placeholder for the future implementation. For now,
        # it will return the first node found.
        # TODO: Implement the disambiguation algorithm
        first_node = get_first_dict_value(choices)
        if not isinstance(first_node, ALLOWED_DRUG_MULTIMAP):
            return choices
        else:
            raise NotImplementedError(
                "Disambiguation of multi-mapped nodes is not implemented yet. "
                "Not even a placeholder."
            )
