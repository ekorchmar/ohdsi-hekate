"""
This module provides NodeOrder class that resolves the order of addition of
different drug and pack classes to the hierarchy.
"""

from __future__ import annotations

from typing import override

import rustworkx as rx  # For graph operations
from rustworkx.visit import BFSVisitor  # For BFS traversal
from rx_model.descriptive.base import (
    ConceptDefinition,
)  # For generic annotation
from rx_model.descriptive.complex import ComplexDrugNodeDefinition  # Drug nodes
from rx_model.descriptive.pack import PackDefinition  # Pack nodes

_SENTINEL = None  # Temporary node to serve as an entry point for BFS


class _BFSSorter[N: ComplexDrugNodeDefinition | PackDefinition](BFSVisitor):
    """
    Class that visits nodes in a breadth-first search manner and records the
    node weights (contents) in order of appearance.
    """

    def __init__(self, graph: ClassHierarchy[N]):
        super().__init__()
        self.visited_node_data: list[N] = []
        self.graph: ClassHierarchy[N] = graph

    @override
    def discover_vertex(self, v: int) -> None:
        payload: N = self.graph[v]  # pyright: ignore[reportAny]
        if payload:  # Not _SENTINEL
            self.visited_node_data.append(payload)


class ClassHierarchy[N: ComplexDrugNodeDefinition | PackDefinition](rx.PyDAG):
    """
    A class that represents the hierarchy of drug and pack classes in a directed
    acyclic graph (DAG) structure.
    """

    def __init__(self):
        super().__init__()
        self.__indices: dict[ConceptDefinition, int] = {}
        self.sentinel_idx: int = self.add_node(_SENTINEL)
        self.bfs_resolver: _BFSSorter[N] = _BFSSorter(self)

    def _add_definition(self, definition: N) -> None:
        """
        Adds a definition node to the graph recursively adding dependencies.
        """
        if definition in self.__indices:
            # Added as a parent
            return

        node_idx = self.add_node(definition)
        self.__indices[definition] = node_idx

        if definition.parent_relations:
            for parent_rel in definition.parent_relations:
                parent_definition = parent_rel.target_definition
                if parent_definition not in self.__indices:
                    self._add_definition(parent_definition)  # pyright: ignore[reportArgumentType]  # noqa: 501
                parent_idx = self.__indices[parent_definition]
                _ = self.add_edge(parent_idx, node_idx, None)
        else:
            # No parent relations, add to sentinel
            _ = self.add_edge(self.sentinel_idx, node_idx, None)

    def populate_from_definitions(self, node_definition: type[N]) -> None:
        """
        Populates the graph with nodes from the provided node definition class.
        """
        for definition in node_definition.registry.values():
            self._add_definition(definition)  # pyright: ignore[reportArgumentType]  # noqa: 501

    def resolve_order(self) -> list[N]:
        """
        Resolves the order of nodes in the graph using BFS traversal.
        """
        rx.bfs_search(self, [self.sentinel_idx], self.bfs_resolver)
        return self.bfs_resolver.visited_node_data

    @classmethod
    def resolve_from_definitions(cls, node_definition: type[N]) -> list[N]:
        """
        Class method to resolve the order of nodes from the provided node
        definition class.
        """
        hierarchy = cls()
        hierarchy.populate_from_definitions(node_definition)
        return hierarchy.resolve_order()
