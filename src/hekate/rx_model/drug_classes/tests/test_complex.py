import pytest
import rx_model.drug_classes.atom as a
import rx_model.drug_classes.complex as c
import rx_model.drug_classes.generic as g
import rx_model.drug_classes.strength as st
from utils.classes import SortedTuple


from .test_atom import (
    permafixture,
    # Ingredients
    acetaminophen,
    paracetamol,
    haloperidol,
    # Precise Ingredients
    haloperidol_decanoate,
    # Brand Names
    advicor,
    bupap,
    # Dosage Forms
    oral_tablet,
    oral_solution,
    # Unit
    mg,
    ml,
)


# CDC
@permafixture
def apap_500mg(acetaminophen, mg):
    return c.ClinicalDrugComponent(
        g.ConceptId(1),
        acetaminophen,
        None,
        st.SolidStrength(500, mg),
    )


@permafixture
def haloperidol_20mg_ml(haloperidol, mg, ml):
    return c.ClinicalDrugComponent(
        g.ConceptId(2),
        haloperidol,
        None,
        st.LiquidConcentration(20, mg, ml),
    )


def test_ingredients_superclasses_of_cdc(
    apap_500mg, haloperidol_20mg_ml, acetaminophen, haloperidol
):
    assert acetaminophen.is_superclass_of(apap_500mg)
    assert haloperidol.is_superclass_of(haloperidol_20mg_ml)
