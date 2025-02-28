"""
Contains the class that hosts the drug concept hierarchy.
"""

from . import drug_classes as dc
import rustworkx as rx


# Generic types
type _AtomicConcept[Id: dc.ConceptIdentifier] = (
    # RxNorm atomic concepts
    dc.Ingredient[Id]
    | dc.DoseForm[Id]
    | dc.BrandName[Id]
    | dc.PreciseIngredient
    |
    # RxNorm Extension atomic concepts
    dc.Supplier[Id]
    |
    # UCUM atomic concepts
    dc.Unit
)
type NumDenomU = tuple[dc.Unit, dc.Unit]
type _UnboundStrength = (
    dc.SolidStrength | dc.LiquidConcentration | dc.LiquidQuantity
)
type _BoundStrength[Id: dc.ConceptIdentifier, S: dc.UnquantifiedStrength] = (
    dc.BoundStrength[Id, S] | dc.BoundQuantity[Id]
)


class Atoms[Id: dc.ConceptIdentifier]:
    """
    Container for atomic concepts like Ingredients and Dose Forms that
    are not guaranteed to participate in the hierarchy.
    """

    def __init__(self):
        self.ingredient: dict[Id, dc.Ingredient[Id]] = {}
        self.dose_form: dict[Id, dc.DoseForm[Id]] = {}
        self.brand_name: dict[Id, dc.BrandName[Id]] = {}
        self.supplier: dict[Id, dc.Supplier[Id]] = {}
        self.unit: dict[Id, dc.Unit] = {}

        # Precise ingredients are stored in a dict from ingredients
        # NOTE: This is not generic, as we do not expect RxE to have precise
        # ingredients. For now.
        self.precise_ingredient: dict[
            dc.Ingredient[dc.ConceptId], list[dc.PreciseIngredient]
        ] = {}

    def add_precise_ingredient(
        self, precise_ingredient: dc.PreciseIngredient
    ) -> None:
        invariant = precise_ingredient.invariant
        self.precise_ingredient.setdefault(invariant, []).append(
            precise_ingredient
        )


class KnownStrength[Id: dc.ConceptIdentifier]:
    """
    Container to store known ingredient strengths and concentrations to avoid
    creating de-facto duplicates with minor differences.

    Strength components are stored in a directed acyclic graph (DAG),
    where roots are ingredient concepts and nodes are strength components.
    """

    def __init__(self):
        self.solid_stength: dict[dc.Unit, dc.SolidStrength] = {}
        self.liquid_concentration: dict[NumDenomU, dc.LiquidConcentration] = {}
        self.liquid_quantity: dict[dc.Unit, dc.LiquidQuantity] = {}
        self.bound_strength_graph: rx.PyDAG = rx.PyDAG(
            check_cycle=False,  # Will be really hard to create a cycle
            multigraph=False,  # Hierarchical structure
            # node_count_hint = 1000,  # TODO: Estimate the number of nodes
            # edge_count_hint = 1000,  # TODO: Estimate the number of edges
        )

    def add_strength(self, strength: _UnboundStrength) -> None:
        del strength
        raise NotImplementedError


class RxHierarchy[Id: dc.ConceptIdentifier]:
    """
    The drug concept hierarchy that contains all the atomic and composite
    concepts.
    """
