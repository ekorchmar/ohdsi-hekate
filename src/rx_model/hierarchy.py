"""
Contains the class that hosts the drug concept hierarchy.
"""

import rx_model.drug_classes as dc
import rustworkx as rx


type _AtomicConcept = (
    # RxNorm atomic concepts
    dc.Ingredient |
    dc.DoseForm |
    dc.BrandName |
    dc.PreciseIngredient |
    # RxNorm Extension atomic concepts
    dc.Supplier |
    # UCUM atomic concepts
    dc.Unit
)
type _Strength = (
    dc.SolidStrength |
    dc.LiquidConcentration |
    dc.LiquidQuantity |
    dc.BoundStrength |
    dc.BoundQuanty
)
type _AtomDict[Id: dc.ConceptIdentifier, RxA: _AtomicConcept] = dict[Id, RxA]


class Atoms[Id: dc.ConceptIdentifier]:
    """
    Container for atomic concepts like Ingredients and Dose Forms that
    are not guaranteed to participate in the hierarchy.
    """

    def __init__(self):
        self.ingredient: _AtomDict[Id, dc.Ingredient] = {}
        self.dose_form: _AtomDict[Id, dc.DoseForm] = {}
        self.brand_name: _AtomDict[Id, dc.BrandName] = {}
        self.precise_ingredient: _AtomDict[Id, dc.PreciseIngredient] = {}
        self.supplier: _AtomDict[Id, dc.Supplier] = {}
        self.unit: _AtomDict[Id, dc.Unit] = {}


class KnownStrength:
    """
    Container to store known ingredient strengths and concentrations to avoid
    creating de-facto duplicates with minor differences.

    Strength components are stored in a directed acyclic graph (DAG),
    where roots are ingredient concepts and nodes are strength components.
    """

    def __init__(self):
        self._strength_graph = rx.PyDAG(
            check_cycle=False,  # Will be really hard to create a cycle
            multigraph=False,  # Hierarchical structure
            # node_count_hint = 1000,  # TODO: Estimate the number of nodes
            # edge_count_hint = 1000,  # TODO: Estimate the number of edges
        )

    def add_strength(self, strength: _Strength):
        raise NotImplementedError


class RxHierarchy:
    """
    The drug concept hierarchy that contains all the atomic and composite
    concepts.
    """
