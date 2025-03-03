"""
Contains the class that hosts the drug concept hierarchy.
"""

from . import drug_classes as dc
import rustworkx as rx
import polars as pl
from typing import Annotated


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
type UnboundStrength = (
    dc.SolidStrength
    | dc.LiquidConcentration
    | dc.LiquidQuantity
    | dc.GaseousPercentage
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

    def add_from_frame(self, frame: pl.DataFrame) -> None:
        """
        Populate the atoms container from a DataFrame.

        Dataframe is expected to have the following columns:
            - concept_id OR concept_code & vocabulary_id combination
            - concept_name
            - concept_class_id
        """
        identifier_columns: list[str]
        identifier_class: type[Id]
        if "concept_id" in frame.columns:
            identifier_columns = ["concept_id"]
            identifier_class = dc.ConceptId  # pyright: ignore[reportAssignmentType] # noqa: E501
        elif {"concept_code", "vocabulary_id"} <= set(frame.columns):
            identifier_columns = ["concept_code", "vocabulary_id"]
            identifier_class = dc.ConceptCodeVocab  # pyright: ignore[reportAssignmentType] # noqa: E501
        else:
            raise ValueError(
                "DataFrame must have either 'concept_id' or both"
                "'concept_code' and 'vocabulary_id' columns."
            )

        # Reorder columns to expected order
        frame = frame.select(
            identifier_columns + ["concept_name", "concept_class_id"]
        )

        # Extract the atoms
        name: str
        cls: str
        identifier: Annotated[list[int], 1] | Annotated[list[str], 2]
        for *identifier, name, cls in frame.iter_rows():
            if cls == "Precise Ingredient":
                continue
            atom_identifier: Id = identifier_class(*identifier)  # pyright: ignore[reportArgumentType] # noqa: E501
            match cls:
                case "Ingredient":
                    container = self.ingredient
                    constructor = dc.Ingredient
                case "Dose Form":
                    container = self.dose_form
                    constructor = dc.DoseForm
                case "Brand Name":
                    container = self.brand_name
                    constructor = dc.BrandName
                case "Supplier":
                    container = self.supplier
                    constructor = dc.Supplier
                case "Unit":
                    container = self.unit
                    constructor = dc.Unit
                case _:
                    raise ValueError(f"Unexpected concept class: {cls}")

            container[atom_identifier] = constructor(  # pyright: ignore[reportArgumentType] # noqa: E501
                identifier=atom_identifier,  # pyright: ignore[reportArgumentType] # noqa: E501
                concept_name=name,
            )

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
        self.gaseous_percentage: dict[dc.Unit, dc.GaseousPercentage] = {}
        self.liquid_quantity: dict[dc.Unit, dc.LiquidQuantity] = {}
        self.bound_strength_graph: rx.PyDAG = rx.PyDAG(
            check_cycle=False,  # Will be really hard to create a cycle
            multigraph=False,  # Hierarchical structure
            # node_count_hint = 1000,  # TODO: Estimate the number of nodes
            # edge_count_hint = 1000,  # TODO: Estimate the number of edges
        )

    def add_strength(self, strength: UnboundStrength) -> None:
        del strength
        raise NotImplementedError


class RxHierarchy[Id: dc.ConceptIdentifier]:
    """
    The drug concept hierarchy that contains all the atomic and composite
    concepts.

    The hierarchy is stored in a directed acyclic graph (DAG), where roots
    are ingredient concepts and nodes are more complex drug concepts.
    Ingredient indices are cached in a dictionary for quick access to entry
    points.
    """

    def __init__(self):
        self.graph: rx.PyDAG = rx.PyDAG(
            check_cycle=False,  # Will be really hard to create a cycle
            multigraph=False,  # Hierarchical structure
            # node_count_hint = 1000,  # TODO: Estimate the number of nodes
            # edge_count_hint = 1000,  # TODO: Estimate the number of edges
        )
        # Cached indices of ingredients (roots)
        self.ingredients: dict[dc.Ingredient[Id], int] = {}

    def add_root(self, root: dc.Ingredient[Id]) -> None:
        """
        Add a root ingredient to the hierarchy.
        """
        self.ingredients[root] = self.graph.add_node(root)

    def add_clinical_drug_form(
        self,
        clinical_drug_form: dc.ClinicalDrugForm[Id],
    ) -> int:
        """
        Add a clinical drug form to the hierarchy. Returns the index of the
        added node in the graph.
        """
        node_idx = self.graph.add_node(clinical_drug_form)
        for ingredient in clinical_drug_form.ingredients:
            _ = self.graph.add_edge(  # Discard edge index
                self.ingredients[ingredient], node_idx, None
            )
        return node_idx
