from rx_model import drug_classes as dc

# Generic types
type AtomicConcept[Id: dc.ConceptIdentifier] = (
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
