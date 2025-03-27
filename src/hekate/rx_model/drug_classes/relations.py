from rx_model.drug_classes.generic import DrugNode, ConceptId
import rx_model.drug_classes.atom as a
import rx_model.drug_classes.strength as st
import rx_model.drug_classes.complex as c

DRUG_CLASS_PREFERENCE_ORDER: list[
    type[DrugNode[ConceptId, st.Strength | None]]
] = [
    c.QuantifiedClinicalBox,
    c.ClinicalDrugBox,
    c.QuantifiedBrandedDrug,
    c.QuantifiedClinicalDrug,
    c.BrandedDrug,
    c.ClinicalDrug,
    c.BrandedDrugForm,
    c.ClinicalDrugForm,
    c.BrandedDrugComponent,
    c.ClinicalDrugComponent,
    a.Ingredient,
]

# Those are intended for isinstance checks, so they are intentionally left
# generic.
ALLOWED_DRUG_MULTIMAP = (
    a.Ingredient,
    c.ClinicalDrugComponent,
)
