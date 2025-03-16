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
    ConceptId,
    DrugNode,
    ForeignDrugNode,
    Ingredient,
    Strength,
)
from rx_model.hierarchy.hosts import NodeIndex, RxHierarchy

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
        # Operands
        self.hierarchy: RxHierarchy[ConceptId] = hierarchy
        self.node: ForeignDrugNode[Strength | None] = node

        self.logger: logging.Logger = logger.getChild(
            f"{self.__class__.__name__}({node.identifier})"
        )

        # The class of nodes to prune the search at. All descendants will be
        # redundant.
        self.aim_for: type[DrugNode[ConceptId, Strength | None]] = (
            self.node.best_case_class()
        )

        # Indicates how many targets of ALLOWED_DRUG_MULTIMAP should
        # the node end up with.
        self.number_components: int = len(self.node.strength_data)

        # Container for the results
        self.terminal_node_indices: set[NodeIndex] = set()
        self.accepted_nodes: set[NodeIndex] = set()

    @override
    def discover_vertex(self, v: NodeIndex) -> None:
        """
        Called when a new vertex is discovered.
        """
        drug_node = self.hierarchy[v]

        if drug_node is self.SENTINEL:
            # Implicitly accept
            return

        if not isinstance(drug_node, Ingredient):
            # Check if all of the node's predecessors were accepted
            if any(
                p_idx not in self.accepted_nodes
                for p_idx in self.hierarchy.predecessor_indices(v)
            ):
                # Not all predecessors are accepted
                raise rx.visit.PruneSearch

        if not drug_node.is_superclass_of(self.node):
            # None of the descendants will match
            raise rx.visit.PruneSearch
        else:
            self._accept_node(v)

    def _accept_node(self, v: NodeIndex) -> None:
        """
        Accepts a node and remembers it.
        """
        drug_node = self.hierarchy[v]
        self.accepted_nodes.add(v)

        # Update the terminal node list, removing the node(s) that is
        # superseded as terminal nodes by the current node.
        self.terminal_node_indices -= {
            idx
            for idx in self.terminal_node_indices
            if self.hierarchy.has_edge(idx, v)
        }
        self.terminal_node_indices.add(v)

        if isinstance(drug_node, self.aim_for):
            # This is the node we are looking for. But we still may
            # match sister nodes, and disambiguation will be needed.
            raise rx.visit.PruneSearch

    def start_search(self) -> None:
        """
        Starts the depth-first search on the hierarchy, starting from the
        ingredient roots.
        """

        self.logger.debug("Starting search for the matching node")

        # NOTE: to make sure that all starting ingredients are found at the same
        # search level, we create an artificial root node that will be the
        # starting point of the search.

        # We make it an Ingredient, so that type checker accepts it
        temporary_root_idx = self.hierarchy.add_node(self.SENTINEL)

        for ing, _ in self.node.get_strength_data():
            ing_idx = self.hierarchy.ingredients[ing]
            _ = self.hierarchy.add_edge(temporary_root_idx, ing_idx, None)

        try:
            rx.bfs_search(self.hierarchy, [temporary_root_idx], self)
        finally:
            # Remove the temporary root node with edges in case we are
            # ever going multithreaded.
            self.hierarchy.remove_node(temporary_root_idx)

    def get_search_results(
        self,
    ) -> dict[NodeIndex, DrugNode[ConceptId, Strength | None]]:
        """
        Returns the search results as a dictionary.
        """
        return {idx: self.hierarchy[idx] for idx in self.terminal_node_indices}
