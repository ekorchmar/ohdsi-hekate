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

    SENTINEL: DrugNode[ConceptId, None] = Ingredient(ConceptId(0), "__SENTINEL")

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
        self.terminal_node_indices: set[NodeIndex] = set()
        self.accepted_nodes: set[NodeIndex] = set()

    @override
    def discover_vertex(self, v: NodeIndex) -> None:
        """
        Called when a new vertex is discovered.
        """
        drug_node = self.hierarchy.graph[v]

        if drug_node is self.SENTINEL:
            # Implicitly accept
            return

        if isinstance(drug_node, Ingredient):
            if self.__matched_ingredient_count >= self.number_components:
                # Stop taking any new ingredient branches
                self._remember_node(v, False)
                raise rx.visit.PruneSearch
        else:
            # Check if the node's predecessors are available in history
            if any(
                p_idx not in self.accepted_nodes
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
        self.accepted_nodes.add(v)

        # Update the terminal node list, removing the node(s) that is
        # superseded as terminal nodes by the current node.
        self.terminal_node_indices -= {
            idx
            for idx in self.terminal_node_indices
            if self.hierarchy.graph.has_edge(idx, v)
        }
        self.terminal_node_indices.add(v)

        if isinstance(drug_node, self.aim_for):
            # This is the node we are looking for. But we still may
            # match sister nodes, and disambiguation will be needed.
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

        # NOTE: to make sure that all starting ingredients are found at the same
        # search level, we create an artificial root node that will be the
        # starting point of the search.

        # We make it an Ingredient, so that type checker accepts it
        temporary_root_idx = self.hierarchy.graph.add_node(self.SENTINEL)

        for ing, _ in self.node.get_strength_data():
            ing_idx = self.hierarchy.ingredients[ing]
            _ = self.hierarchy.graph.add_edge(temporary_root_idx, ing_idx, None)

        try:
            rx.bfs_search(self.hierarchy.graph, [temporary_root_idx], self)
        finally:
            # Remove the temporary root node with edges in case we are
            # ever going multithreaded.
            self.hierarchy.graph.remove_node(temporary_root_idx)

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
        if not self.terminal_node_indices:
            raise ValueError("No nodes found. Did you call start_search()?")

        msg = f"Found {len(self.terminal_node_indices)} best case nodes:"
        for idx in sorted(self.terminal_node_indices):
            msg += f"\n - {self.hierarchy.graph[idx]}"
        self.logger.debug(msg)

        return {
            idx: self.hierarchy.graph[idx] for idx in self.terminal_node_indices
        }

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
