import rx_model.drug_classes.atom as a
import rx_model.drug_classes.complex as c
import rx_model.drug_classes.pack as p

# TODO: move to "descriptive" module

DRUG_CLASS_PREFERENCE_ORDER = [
    c.QuantifiedBrandedBox,
    c.QuantifiedClinicalBox,
    c.BrandedDrugBox,
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

PACK_CLASS_PREFERENCE_ORDER = [
    p.BrandedPackBox,
    p.BrandedPack,
    p.ClinicalPackBox,
    p.ClinicalPack,
]
