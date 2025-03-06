from typing import Final

import rx_model.drug_classes.atom as a
import rustworkx as rx
from rx_model.drug_classes.complex import (
    BrandedDrug,
    BrandedDrugComponent,
    BrandedDrugForm,
    ClinicalDrug,
    ClinicalDrugComponent,
    ClinicalDrugForm,
    QuantifiedBrandedDrug,
    QuantifiedClinicalDrug,
)
from rx_model.drug_classes.generic import ConceptIdentifier, DrugNode

DRUG_CLASS_DEPENDENCY: rx.PyDiGraph[type[DrugNode[ConceptIdentifier]], None] = (
    rx.PyDiGraph(node_count_hint=11, edge_count_hint=121)
)

_indices: dict[type[DrugNode[ConceptIdentifier]], int] = {}
# Add all classes to the dependency graph
for class_ in [
    a.Ingredient,
    ClinicalDrugComponent,
    BrandedDrugComponent,
    ClinicalDrugForm,
    BrandedDrugForm,
    ClinicalDrug,
    BrandedDrug,
    QuantifiedClinicalDrug,
    QuantifiedBrandedDrug,
]:
    _indices[class_] = DRUG_CLASS_DEPENDENCY.add_node(class_)

DRUG_CLASS_DEPENDENCY.extend_from_edge_list(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # noqa: E501
    list(
        map(
            lambda edge: (_indices[edge[0]], _indices[edge[1]]),
            [
                (a.Ingredient, ClinicalDrugComponent),
                (a.Ingredient, ClinicalDrugForm),
                (ClinicalDrugComponent, ClinicalDrug),
                (ClinicalDrugComponent, BrandedDrugComponent),
                (ClinicalDrugForm, ClinicalDrug),
                (ClinicalDrugForm, BrandedDrugForm),
                (ClinicalDrug, QuantifiedClinicalDrug),
                (BrandedDrugComponent, BrandedDrug),
                (BrandedDrugForm, BrandedDrug),
                (QuantifiedClinicalDrug, QuantifiedBrandedDrug),
                (BrandedDrug, QuantifiedBrandedDrug),
            ],
        )
    )
)

DRUG_CLASS_PREFERENCE_ORDER = [
    QuantifiedBrandedDrug,
    QuantifiedClinicalDrug,
    BrandedDrug,
    ClinicalDrug,
    BrandedDrugComponent,
    ClinicalDrugComponent,
    BrandedDrugForm,
    ClinicalDrugForm,
    a.Ingredient,
]

# Those are intended for isinstance checks, so they are intentionally left
# generic.
ALLOWED_DRUG_MULTIMAP: Final[tuple[type[DrugNode], ...]] = (  # pyright: ignore[reportMissingTypeArgument]  # noqa: E501
    a.Ingredient,
    ClinicalDrugComponent,
)
