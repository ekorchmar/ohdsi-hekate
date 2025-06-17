"""
Implementations of the Marketed Product pseudo-class.
"""

from dataclasses import dataclass
from enum import Enum  # For defining terminal parent types
from typing import override  # for HierarchyNode interface

from rx_model.drug_classes import complex as c  # For content type verification
from rx_model.drug_classes import pack as p  # For content type verification
from rx_model.drug_classes.atom import (
    Ingredient,  # For superclass checks
    Supplier,  # For defining attribute
)
from rx_model.drug_classes.base import (
    ConceptIdentifier,  # For identifiers
    HierarchyNode,  # Inheriting
)
from rx_model.drug_classes.generic import (
    DrugNode,  # Drug content
    PackNode,  # Pack content
)
from rx_model.drug_classes import strength as st  # For strength data
from utils.exceptions import RxConceptCreationError


class MarketedProductTerminalParent(Enum):
    """
    Enum to define the type of marketed product.
    This is a subset of ConceptClassId that represents marketed products.
    """

    CD = c.ClinicalDrug
    QCD = c.QuantifiedClinicalDrug
    BD = c.BrandedDrug
    QBD = c.QuantifiedBrandedDrug
    CDB = c.ClinicalDrugBox
    BDB = c.BrandedDrugBox
    QCB = c.QuantifiedClinicalBox
    QBB = c.QuantifiedBrandedBox
    BP = p.BrandedPack
    CP = p.ClinicalPack


@dataclass(frozen=True, order=True, eq=True, slots=True)
class MarketedProductNode[Id: ConceptIdentifier](HierarchyNode[Id]):
    """
    Represents a Marketed Product in the RxNorm hierarchy.
    Inherits from HierarchyNode to provide structure and relationships.
    """

    identifier: Id
    terminal_parent: (
        DrugNode[Id, st.SolidStrength | st.LiquidQuantity] | PackNode[Id]
    )
    supplier: Supplier[Id]

    @override
    def is_superclass_of(
        self, other: HierarchyNode[Id], passed_hierarchy_checks: bool = True
    ) -> bool:
        """
        Check if this node is a superclass of another node.

        Args:
            other: The node to check against.
            passed_hierarchy_checks: Whether the tested node had already passed
                the corresponding checks by the predecessors of this node. This
                is used to avoid redundant checks in the hierarchy.
        """

        if passed_hierarchy_checks or self.terminal_parent.is_superclass_of(
            other, passed_hierarchy_checks
        ):
            # Require supplier match
            match other:
                case Ingredient():
                    # No supplier in Ingredient
                    return False
                case DrugNode() | PackNode():
                    return self.supplier == other.get_supplier()
                case MarketedProductNode():
                    return self.supplier == other.supplier
                case _:
                    # Unreachable
                    raise ValueError(
                        f"{other} is not a valid node type to test."
                    )
        return False

    def __post_init__(self):
        """
        Post-initialization to ensure the terminal parent is a valid DrugNode or
        PackNode.
        """
        # Make sure that the terminal parent belongs to allowed types
        if not isinstance(
            self, tuple(MarketedProductTerminalParent._value2member_map_)
        ):
            raise RxConceptCreationError(
                f"Invalid terminal parent type: {type(self.terminal_parent)}"
            )

        # For DrugNode, make sure the strength is not a concentration
        if isinstance(self.terminal_parent, DrugNode):
            for _, strength in self.terminal_parent.get_strength_data():
                if isinstance(
                    strength, (st.LiquidConcentration, st.GasPercentage)
                ):
                    raise RxConceptCreationError(
                        f"Invalid strength data for DrugNode: "
                        f"{strength.__class__.__name__}"
                    )
