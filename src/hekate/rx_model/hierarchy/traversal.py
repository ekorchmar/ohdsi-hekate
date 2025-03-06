"""
Hosts a visitor class that traverses the RxHierarchy tree (BFS) and finds a new
appropriate location for a new node.

Note: PruneSearch is raised to prune the search tree while traversing a graph.
It will not stop the search, only discard the successor nodes.
"""

from collections import defaultdict
from typing import override

import rustworkx as rx
from rx_model.drug_classes import (
    ALLOWED_DRUG_MULTIMAP,
    ConceptIdentifier,
    DrugNode,
    ForeignDrugNode,
    Ingredient,
)
from rx_model.hierarchy.hosts import HierarchyChecksum, NodeIndex, RxHierarchy

type NodeAcceptance = bool


class DrugNodeFinder[Id: ConceptIdentifier](rx.visit.BFSVisitor):
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
        self.history: dict[  # Unused for now
            HierarchyChecksum, dict[NodeIndex, NodeAcceptance]
        ] = defaultdict(dict)

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

        self.current_hierarchy: RxHierarchy[Id] | None = None

        # Container for the results
        self.final_nodes: dict[NodeIndex, DrugNode[Id]] = {}

    @override
    def discover_vertex(self, v: NodeIndex) -> None:
        """
        Called when a new vertex is discovered.
        """
        if self.current_hierarchy is None:
            raise ValueError(
                "Hierarchy not set. Only start iteration with start_search() "
                "method."
            )

        drug_node: DrugNode[Id] = self.current_hierarchy.graph[v]

        if isinstance(drug_node, Ingredient):
            if self.__matched_ingredient_count >= self.number_components:
                # Stop taking any new ingredient branches
                self._remember_node(v, False)
                raise rx.visit.PruneSearch

        acceptance: NodeAcceptance = drug_node.is_superclass_of(self.node)
        self._remember_node(v, acceptance)

        if not acceptance:
            # None of the descendants will match
            raise rx.visit.PruneSearch
        else:
            self.final_nodes[v] = drug_node
            if isinstance(drug_node, self.aim_for):
                # We are almost there. However, this might be a node for
                # multi-mapping.
                can_be_multi = self.number_components > 1 and isinstance(
                    drug_node, ALLOWED_DRUG_MULTIMAP
                )

                if can_be_multi:
                    raise NotImplementedError(
                        f"Multi-mapping is not implemented yet. However, the "
                        f"node {self.node.identifier} can be mapped to the "
                        f"node {drug_node.identifier}."
                    )
                else:
                    # This is the node we are looking for. But we still may
                    # match sister nodes.
                    raise rx.visit.PruneSearch

    def _remember_node(self, v: NodeIndex, acceptance: NodeAcceptance) -> None:
        """
        Remembers the acceptance of a node.
        """
        assert self.current_hierarchy is not None
        self.history[self.current_hierarchy.get_checksum()][v] = acceptance

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

        self.__matched_ingredient_count = 0

        self.current_hierarchy = hierarchy
        rx.bfs_search(
            hierarchy.graph, list(hierarchy.ingredients.values()), self
        )

    def get_search_results(self) -> dict[NodeIndex, DrugNode[Id]]:
        """
        Returns the final nodes that have been found.
        """
        return self.final_nodes
