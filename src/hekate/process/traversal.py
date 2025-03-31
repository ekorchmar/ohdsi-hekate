"""
Hosts a visitor class that traverses the RxHierarchy tree (BFS) and finds a new
appropriate location for a new node.

Note: PruneSearch is raised to prune the search tree while traversing a graph.
It will not stop the search, only discard the successor nodes.
"""

import importlib.util  # For conditional imports
import logging  # Typing
from pathlib import Path
from typing import (
    NoReturn,
    override,  # To mark the overridden BFSVisitor methods
)

import rustworkx as rx  # Core implementation dependency
from rustworkx.visualization import (  # For illustration
    graphviz_draw,  # pyright: ignore[reportUnknownVariableType]
)
from rx_model.drug_classes import (
    ConceptId,  # Typing
    ConceptCodeVocab,  # Typing
    DrugNode,  # Operand
    ForeignDrugNode,  # Operand
    Ingredient,  # Entry point, special case
    Strength,  # Typing
)
from rx_model.hierarchy.hosts import NodeIndex, RxHierarchy


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
        save_subplot: bool = False,
    ):
        """
        Initializes the visitor with the hierarchy and the new node to be
        tested.

        Args:
            node: The nativized representation of the source node as predicate.
            hierarchy: The target RxHierarchy to search in.
            logger: The logger to get a child logger from for the visitor.
            save_subplot: Whether to save the subgraph of the hierarchy that
                was traversed. Useful for debugging.
        """
        # Operands
        self.hierarchy: RxHierarchy[ConceptId] = hierarchy
        self.node: ForeignDrugNode[Strength | None] = node

        self.logger: logging.Logger = logger.getChild(
            f"{self.__class__.__name__}({node.identifier})"
        )

        # The class of nodes to prune the search at. All descendants will be
        # redundant.
        self.aim_for: type[DrugNode] = (  # pyright: ignore[reportMissingTypeArgument]  # noqa: E501
            self.node.best_case_class()
        )

        # Indicates how many targets of ALLOWED_DRUG_MULTIMAP should
        # the node end up with.
        self.number_components: int = len(self.node.strength_data)

        # Container for the results
        self.terminal_node_indices: set[NodeIndex] = set()
        self.accepted_nodes: set[NodeIndex] = set()

        # Whether to save the subgraph of the hierarchy that was traversed.
        self.save_subgraph: bool = save_subplot
        # Will only be populated if save_subplot is True
        self.rejected_nodes: set[NodeIndex] = set()
        self._rejected_by_accepted: dict[NodeIndex, set[NodeIndex]] = {}
        self.subgraph: (
            None | rx.PyDiGraph[DrugNode[ConceptId, Strength | None], None]
        ) = None

    @override
    def discover_vertex(self, v: NodeIndex) -> None:
        """
        Called when a new vertex is discovered.
        """
        drug_node = self.hierarchy[v]

        if drug_node is self.SENTINEL:
            # Implicitly accept
            return

        if not isinstance(drug_node, Ingredient):  # pyright: ignore[reportUnnecessaryIsInstance]  # noqa: E501
            # Check if all of the node's predecessors were accepted
            if any(
                p_idx not in self.accepted_nodes
                for p_idx in self.hierarchy.predecessor_indices(v)
            ):
                # Not all predecessors are accepted
                self._reject_node(v)

        if not drug_node.is_superclass_of(self.node):
            # None of the descendants will match
            self._reject_node(v)
        else:
            self._accept_node(v)

    def _reject_node(self, v: NodeIndex) -> NoReturn:
        """
        Rejects a node and remembers it, if needed; search is pruned.
        """
        if self.save_subgraph:
            self.rejected_nodes.add(v)
        raise rx.visit.PruneSearch

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

        if isinstance(drug_node, self.aim_for):  # pyright: ignore[reportUnknownMemberType]  # noqa: E501
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
        temporary_root_idx = self.hierarchy.add_node(self.SENTINEL)  # pyright: ignore[reportArgumentType]  # noqa: E501

        for ing, _ in self.node.get_strength_data():
            ing_idx = self.hierarchy.ingredients[ing]
            _ = self.hierarchy.add_edge(temporary_root_idx, ing_idx, None)

        try:
            rx.bfs_search(self.hierarchy, [temporary_root_idx], self)
        finally:
            # Remove the temporary root node even if traversal fails
            self.hierarchy.remove_node(temporary_root_idx)

    def _color_subplot_edges(self) -> None:
        """
        Colors the edges of the subgraph to indicate the traversal result.
        """

    def get_search_results(
        self,
    ) -> dict[NodeIndex, DrugNode[ConceptId, Strength | None]]:
        """
        Returns the search results as a dictionary.
        """
        return {idx: self.hierarchy[idx] for idx in self.terminal_node_indices}

    def draw_subgraph(
        self,
        save_path: Path,
        use_identifier: ConceptCodeVocab,
    ) -> None:
        """
        Illustrates the subgraph that was traversed.

        Args:
            save_path: The path to save the illustration to.
            use_identifier: The identifier to use to label the graph.
        """
        if not all(
            importlib.util.find_spec(mod) for mod in ["PIL", "graphviz"]
        ):
            self.logger.warning(
                "Cannot illustrate the subgraph, missing dependencies"
            )
            return

        if not self.accepted_nodes:
            self.logger.warning("No nodes were accepted, nothing to illustrate")
            return

        if self.subgraph is None:
            self.subgraph = self.hierarchy.subgraph(list(self.accepted_nodes))

        self._rejected_by_accepted = {}
        for source, target in self.hierarchy.edge_list():
            if source in self.accepted_nodes and target in self.rejected_nodes:
                self._rejected_by_accepted.setdefault(source, set()).add(target)

        label = (
            f"Visited nodes for {use_identifier}; "
            f"accepted: {len(self.accepted_nodes)}, "
            f"rejected: {len(self.rejected_nodes)}"
        )
        _ = graphviz_draw(  # pyright: ignore[reportUnknownVariableType]
            self.subgraph,
            node_attr_fn=self._get_graphviz_node_attr,
            graph_attr={"label": '"' + label + '"'},
            filename=str(save_path),
            image_type="svg",
        )

    def _get_graphviz_node_attr(
        self, node: DrugNode[ConceptId, Strength | None]
    ) -> dict[str, str]:
        """
        Returns the Graphviz node attributes for the given node.
        """
        # PERF: We are passing the node data as a parameter, not index; but we
        # need index
        node_index = next(
            idx for idx in self.accepted_nodes if self.hierarchy[idx] is node
        )
        label = f"{node.identifier} ({node.__class__.__name__})"
        rejected_subtypes = len(self._rejected_by_accepted.get(node_index, []))

        return {
            "label": label + f"\\nRejected: {rejected_subtypes} successors"
            if rejected_subtypes
            else label,
        }
