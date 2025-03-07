import rx_model.drug_classes.atom as a
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

DRUG_CLASS_PREFERENCE_ORDER = [
    QuantifiedBrandedDrug,
    QuantifiedClinicalDrug,
    BrandedDrug,
    ClinicalDrug,
    BrandedDrugForm,
    ClinicalDrugForm,
    BrandedDrugComponent,
    ClinicalDrugComponent,
    a.Ingredient,
]

# Those are intended for isinstance checks, so they are intentionally left
# generic.
ALLOWED_DRUG_MULTIMAP = (
    a.Ingredient,
    ClinicalDrugComponent,
)
