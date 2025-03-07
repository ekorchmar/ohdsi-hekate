"""
Hosts a visitor class that traverses the RxHierarchy tree (BFS) and finds a new
appropriate location for a new node.

Note: PruneSearch is raised to prune the search tree while traversing a graph.
It will not stop the search, only discard the successor nodes.
"""

from typing import override

import rustworkx as rx
from rx_model.drug_classes import (
    ALLOWED_DRUG_MULTIMAP,
    ConceptIdentifier,
    DrugNode,
    ForeignDrugNode,
    Ingredient,
)
from rx_model.hierarchy.hosts import NodeIndex, RxHierarchy

type NodeAcceptance = bool


class DrugNodeFinder[Id: ConceptIdentifier](rx.visit.BFSVisitor):
    """
    A visitor class that traverses the RxHierarchy tree (BFS) and finds a new
    appropriate location for a new node.
    """

    def __init__(self, node: ForeignDrugNode[Id], hierarchy: RxHierarchy[Id]):
        """
        Initializes the visitor with the hierarchy and the new node to be
        tested.
        """
        self.node: ForeignDrugNode[Id] = node
        self.history: dict[NodeIndex, NodeAcceptance] = {}

        self.aim_for: type[DrugNode[Id]] = self.node.best_case_class()

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
        self.final_nodes: dict[NodeIndex, DrugNode[Id]] = {}

    @override
    def discover_vertex(self, v: NodeIndex) -> None:
        """
        Called when a new vertex is discovered.
        """
        drug_node: DrugNode[Id] = self.hierarchy.graph[v]

        if isinstance(drug_node, Ingredient):
            if self.__matched_ingredient_count >= self.number_components:
                # Stop taking any new ingredient branches
                self._remember_node(v, False)
                raise rx.visit.PruneSearch
        else:
            if not self._accepted_all_predecessors(v):
                self._remember_node(v, False)
                raise rx.visit.PruneSearch

        acceptance: NodeAcceptance = drug_node.is_superclass_of(self.node)
        self._remember_node(v, acceptance)

        if not acceptance:
            # None of the descendants will match
            raise rx.visit.PruneSearch
        else:
            self._accept_node(v)

    def _accepted_all_predecessors(self, v: NodeIndex) -> bool:
        """
        Checks if all predecessors of a node have been accepted.
        """

        # Because it is a BFS, all predecessors should be already visited
        return any(
            p_idx not in self.final_nodes
            for p_idx in self.hierarchy.graph.predecessor_indices(v)
        )

    def _accept_node(self, v: NodeIndex) -> None:
        """
        Accepts a node and remembers it.
        """
        drug_node: DrugNode[Id] = self.hierarchy.graph[v]
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

        self.__matched_ingredient_count = 0

        rx.bfs_search(
            self.hierarchy.graph,
            list(self.hierarchy.ingredients.values()),
            self,
        )

    def get_search_results(self) -> dict[NodeIndex, DrugNode[Id]]:
        """
        Returns the final nodes that have been found.
        """
        return self.final_nodes
