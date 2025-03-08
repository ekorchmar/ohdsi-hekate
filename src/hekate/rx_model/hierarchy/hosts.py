"""
Contains the classes that hosts the drug concept hierarchy.
"""

import logging
from rx_model.hierarchy.generic import NumDenomU, AtomicConcept
from collections.abc import Iterable
from collections import ChainMap
from typing import Annotated

import polars as pl
import rustworkx as rx
from rx_model import drug_classes as dc
from utils.exceptions import InvalidConceptIdError

type HierarchyChecksum = int
type NodeIndex = int


class Atoms[Id: dc.ConceptIdentifier]:
    """
    Container for atomic concepts like Ingredients and Dose Forms that
    are not guaranteed to participate in the hierarchy.
    """

    def lookup_unknown(self, identifier: Id) -> AtomicConcept[Id]:
        """
        Lookup an atomic concept by its identifier. Concept class is not
        guaranteed to be known.

        This method should only be used by a caller that operates on verifyable
        data.
        """
        try:
            return ChainMap(
                self.ingredient,
                self.dose_form,  # pyright: ignore[reportArgumentType]
                self.brand_name,  # pyright: ignore[reportArgumentType]
                self.supplier,  # pyright: ignore[reportArgumentType]
            )[identifier]
        except KeyError:
            raise InvalidConceptIdError(
                f"Concept ID {identifier} not found in Atoms container."
            )

    def __init__(self, logger: logging.Logger):
        self.ingredient: dict[Id, dc.Ingredient[Id]] = {}
        self.dose_form: dict[Id, dc.DoseForm[Id]] = {}
        self.brand_name: dict[Id, dc.BrandName[Id]] = {}
        self.supplier: dict[Id, dc.Supplier[Id]] = {}
        self.unit: dict[Id, dc.Unit] = {}

        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info("Initialized Atoms container.")

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

        self.logger.info(f"Added {len(frame)} atoms to the container.")

    def add_precise_ingredient(
        self, precise_ingredient: dc.PreciseIngredient
    ) -> None:
        invariant = precise_ingredient.invariant
        self.precise_ingredient.setdefault(invariant, []).append(
            precise_ingredient
        )


class KnownStrengths[Id: dc.ConceptIdentifier]:
    """
    Container to store known ingredient strengths and concentrations to avoid
    creating de-facto duplicates with minor differences.

    Strength components are stored in a directed acyclic graph (DAG),
    where roots are ingredient concepts and nodes are strength components.
    """

    def __init__(self):
        # Associate strength components with units
        self.solid_stength: dict[dc.Unit, dc.SolidStrength] = {}
        self.liquid_concentration: dict[NumDenomU, dc.LiquidConcentration] = {}
        self.gaseous_percentage: dict[dc.Unit, dc.GasPercentage] = {}
        self.liquid_quantity: dict[dc.Unit, dc.LiquidQuantity] = {}

        # Associate strength components with ingredients
        self.known_associations_graph: dict[dc.Ingredient[Id], dc.Strength] = {}
        self.ingredients: dict[dc.Ingredient[Id], NodeIndex] = {}

    def add_strength(self, strength: dc.Strength) -> None:
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

    def __init__(self, logger):
        self.graph: rx.PyDAG[dc.DrugNode[Id, dc.Strength | None], None] = (
            rx.PyDAG(
                check_cycle=False,  # Will be really hard to create a cycle
                multigraph=False,  # Hierarchical structure
                # node_count_hint = 1000,  # TODO: Estimate the number of nodes
                # edge_count_hint = 1000,  # TODO: Estimate the number of edges
            )
        )
        # Cached indices of ingredients (roots)
        self.ingredients: dict[dc.Ingredient[Id], NodeIndex] = {}
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info("Initialized RxHierarchy.")

    def add_root(self, root: dc.Ingredient[Id]) -> None:
        """
        Add a root ingredient to the hierarchy.
        """
        self.ingredients[root] = self.graph.add_node(root)

    def add_clinical_drug_form(
        self,
        clinical_drug_form: dc.ClinicalDrugForm[Id],
    ) -> NodeIndex:
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

    def add_clinical_drug_component[S: dc.UnquantifiedStrength](
        self,
        clinical_drug_component: dc.ClinicalDrugComponent[Id, S],
    ) -> NodeIndex:
        """
        Add a clinical drug component to the hierarchy. Returns the index of
        the added node in the graph.
        """
        # TODO: add strength data

        node_idx = self.graph.add_node(clinical_drug_component)
        _ = self.graph.add_edge(  # Discard edge index
            self.ingredients[clinical_drug_component.ingredient],
            node_idx,
            None,
        )
        return node_idx

    def add_branded_drug_form(
        self,
        branded_drug_form: dc.BrandedDrugForm[Id],
        clinical_drug_form_idx: NodeIndex,
    ) -> NodeIndex:
        """
        Add a branded drug form to the hierarchy. Returns the index of the
        added node in the graph.
        """
        node_idx = self.graph.add_node(branded_drug_form)
        _ = self.graph.add_edge(  # Discard edge index
            clinical_drug_form_idx, node_idx, None
        )
        return node_idx

    def add_branded_drug_component[S: dc.UnquantifiedStrength](
        self,
        branded_drug_component: dc.BrandedDrugComponent[Id, S],
        clinical_drug_component_idxs: Iterable[NodeIndex],
    ) -> NodeIndex:
        """
        Add a branded drug component to the hierarchy. Returns the index of
        the added node in the graph.
        """
        node_idx = self.graph.add_node(branded_drug_component)
        for cdc_idx in clinical_drug_component_idxs:
            _ = self.graph.add_edge(  # Discard edge index
                cdc_idx, node_idx, None
            )
        return node_idx

    def add_clinical_drug[S: dc.UnquantifiedStrength](
        self,
        clinical_drug: dc.ClinicalDrug[Id, S],
        clinical_drug_form_idx: NodeIndex,
        clinical_drug_component_idxs: Iterable[NodeIndex],
    ) -> NodeIndex:
        """
        Add a clinical drug to the hierarchy. Returns the index of the added
        node in the graph.
        """
        node_idx = self.graph.add_node(clinical_drug)
        _ = self.graph.add_edge(  # Discard edge index
            clinical_drug_form_idx, node_idx, None
        )
        for cdc_idx in clinical_drug_component_idxs:
            _ = self.graph.add_edge(  # Discard edge index
                cdc_idx, node_idx, None
            )
        return node_idx

    def add_branded_drug[S: dc.UnquantifiedStrength](
        self,
        branded_drug: dc.BrandedDrug[Id, S],
        clinical_drug_idx: NodeIndex,
        branded_drug_form_idx: NodeIndex,
        branded_drug_component_idx: NodeIndex,
    ) -> NodeIndex:
        """
        Add a branded drug to the hierarchy. Returns the index of the added
        node in the graph.
        """
        node_idx = self.graph.add_node(branded_drug)
        for idx in [
            clinical_drug_idx,
            branded_drug_form_idx,
            branded_drug_component_idx,
        ]:
            _ = self.graph.add_edge(  # Discard edge index
                idx, node_idx, None
            )
        return node_idx

    def add_quantified_clinical_drug(
        self,
        quantified_clinical_drug: dc.QuantifiedClinicalDrug[
            Id, dc.LiquidConcentration | dc.GasPercentage
        ],
        clinical_drug_idx: NodeIndex,
    ) -> NodeIndex:
        """
        Add a quantified clinical drug to the hierarchy. Returns the index of
        the added node in the graph.
        """
        node_idx = self.graph.add_node(quantified_clinical_drug)
        _ = self.graph.add_edge(  # Discard edge index
            clinical_drug_idx, node_idx, None
        )
        return node_idx

    def add_quantified_branded_drug(
        self,
        quantified_branded_drug: dc.QuantifiedBrandedDrug[Id],
        branded_drug_idx: NodeIndex,
        quantified_clinical_drug_idx: NodeIndex,
    ) -> NodeIndex:
        """
        Add a quantified branded drug to the hierarchy. Returns the index of
        the added node in the graph.
        """

        node_idx = self.graph.add_node(quantified_branded_drug)
        for idx in [
            branded_drug_idx,
            quantified_clinical_drug_idx,
        ]:
            _ = self.graph.add_edge(  # Discard edge index
                idx, node_idx, None
            )
        return node_idx

    def get_checksum(self) -> HierarchyChecksum:
        """
        Get a checksum of the hierarchy. This is useful to verify the integrity
        of the hierarchy after serialization and deserialization, as well as
        for dependent classes to be able to associate data with a certain state
        of the hierarchy.
        """
        # WARN: This is not really implemented yet.
        self.logger.warning(
            "Checksum calculation is not implemented. Returning 0."
        )
        return 0
