"""
Hosts a visitor class that traverses the RxHierarchy tree (BFS) and finds a new
appropriate location for a new node.

Note: PruneSearch is raised to prune the search tree while traversing a graph.
It will not stop the search, only discard the successor nodes.
"""

from typing import override
from rx_model.drug_classes import ForeignDrugNode, ConceptIdentifier
from rx_model.hierarchy.hosts import RxHierarchy, HierarchyChecksum, NodeIndex
import rustworkx as rx
from collections import defaultdict


type NodeAcceptance = bool


class DrugNodeTester[Id: ConceptIdentifier](rx.visit.BFSVisitor):
    """
    A visitor class that traverses the RxHierarchy tree (BFS) and finds a new
    appropriate location for a new node.
    """

    def __init__(self, node: ForeignDrugNode[Id]):
        """
        Initializes the visitor with the hierarchy and the new node to be
        tested.
        """
        self.node: ForeignDrugNode[Id] = node
        self.history: dict[
            HierarchyChecksum, dict[NodeIndex, NodeAcceptance]
        ] = defaultdict(dict)

    @override
    def discover_vertex(self, v: int) -> None:
        """
        Called when a new vertex is discovered.
        """
        raise NotImplementedError("Not implemented yet.")

    def start_search(self, hierarchy: RxHierarchy[Id]) -> None:
        """
        Starts the depth-first search on the hierarchy, starting from the
        ingredient roots.
        """
        if hierarchy.get_checksum() in self.history:
            raise NotImplementedError(
                "Hierarchy has already been searched and continuation is not "
                "implemented yet."
            )
        rx.bfs_search(
            hierarchy.graph, list(hierarchy.ingredients.values()), self
        )
