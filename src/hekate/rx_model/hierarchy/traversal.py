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
    ConceptIdentifier,
    DrugNode,
    ForeignDrugNode,
    Ingredient,
    Strength,
)
from rx_model.hierarchy.hosts import NodeIndex, RxHierarchy

type NodeAcceptance = bool


class DrugNodeFinder[Id: ConceptIdentifier](rx.visit.BFSVisitor):
    """
    A visitor class that traverses the RxHierarchy tree (BFS) and finds a new
    appropriate location for a new node.
    """

    def __init__(
        self,
        node: ForeignDrugNode[Id, Strength | None],
        hierarchy: RxHierarchy[Id],
        logger: logging.Logger,
    ):
        """
        Initializes the visitor with the hierarchy and the new node to be
        tested.
        """
        self.node: ForeignDrugNode[Id, Strength | None] = node
        self.logger = logger.getChild(
            f"{self.__class__.__name__}({node.identifier})"
        )

        self.history: dict[NodeIndex, NodeAcceptance] = {}

        self.aim_for: type[DrugNode[Id, Strength | None]] = (
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

        self.hierarchy: RxHierarchy[Id] = hierarchy

        # Container for the results
        self.final_nodes: dict[NodeIndex, DrugNode[Id, Strength | None]] = {}

    @override
    def discover_vertex(self, v: NodeIndex) -> None:
        """
        Called when a new vertex is discovered.
        """
        drug_node: DrugNode[Id, Strength | None] = self.hierarchy.graph[v]

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
        drug_node: DrugNode[Id, Strength | None] = self.hierarchy.graph[v]
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

        rx.bfs_search(
            self.hierarchy.graph,
            list(self.hierarchy.ingredients.values()),
            self,
        )

    def get_search_results(
        self,
    ) -> dict[NodeIndex, DrugNode[Id, Strength | None]]:
        """
        Returns NOT DISAMBIGUATED best case nodes found during the search.
        """
        if not self.final_nodes:
            raise ValueError("No nodes found. Did you call start_search()?")

        best_nodes = reversed(self.final_nodes)
        best_node_idx = next(best_nodes)
        best_node = self.final_nodes[best_node_idx]

        best_class_: type[DrugNode[Id, Strength | None]]
        for best_class_ in DRUG_CLASS_PREFERENCE_ORDER:
            if isinstance(best_node, best_class_):
                break
        else:
            raise ValueError(
                f"Encountered unexpected type {best_node.__class__.__name__} "
                f"which node {best_node.identifier} is an instance of."
            )

        best_nodes = {best_node_idx: best_node}
        for idx in best_nodes:
            node = best_nodes[idx]
            if isinstance(node, best_class_):
                best_nodes[idx] = node

        self.logger.debug(
            f"Found {len(best_nodes)} best case nodes: {best_nodes}"
        )
        return best_nodes
