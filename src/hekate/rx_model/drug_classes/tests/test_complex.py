import pytest
import rx_model.drug_classes.atom as a
import rx_model.drug_classes.complex as c
import rx_model.drug_classes.base as b
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
    haloperidol_etherate,
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
        b.ConceptId(1),
        acetaminophen,
        None,
        st.SolidStrength(500, mg),
    )


@permafixture
def apap_40mg_ml(acetaminophen, mg):
    return c.ClinicalDrugComponent(
        b.ConceptId(1),
        acetaminophen,
        None,
        st.LiquidConcentration(40, mg, ml),
    )


@permafixture
def haloperidol_40mg_ml(haloperidol, mg, ml):
    return c.ClinicalDrugComponent(
        b.ConceptId(2),
        haloperidol,
        None,
        st.LiquidConcentration(40, mg, ml),
    )


@permafixture
def haloperidol_40mg_ml(haloperidol, mg, ml):
    return c.ClinicalDrugComponent(
        b.ConceptId(2),
        haloperidol,
        None,
        st.LiquidConcentration(40, mg, ml),
    )


@permafixture
def haloperidol_decanoate_40mg_ml(haloperidol, mg, ml, haloperidol_decanoate):
    return c.ClinicalDrugComponent(
        b.ConceptId(2),
        haloperidol,
        haloperidol_decanoate,
        st.LiquidConcentration(40, mg, ml),
    )


@permafixture
def haloperidol_etherate_40mg_ml(haloperidol, mg, ml, haloperidol_etherate):
    return c.ClinicalDrugComponent(
        b.ConceptId(2),
        haloperidol,
        haloperidol_etherate,
        st.LiquidConcentration(40, mg, ml),
    )


@permafixture
def _haloperidol_39mg_ml(haloperidol, mg, ml):
    return c.ClinicalDrugComponent(
        b.ConceptId(2),
        haloperidol,
        None,
        st.LiquidConcentration(39, mg, ml),
    )


@permafixture
def _haloperidol_41mg_ml(haloperidol, mg, ml):
    return c.ClinicalDrugComponent(
        b.ConceptId(2),
        haloperidol,
        None,
        st.LiquidConcentration(41, mg, ml),
    )


@permafixture
def _haloperidol_45mg_ml(haloperidol, mg, ml):
    return c.ClinicalDrugComponent(
        b.ConceptId(2),
        haloperidol,
        None,
        st.LiquidConcentration(45, mg, ml),
    )


def test_ingredients_superclasses_of_cdc(
    apap_500mg, haloperidol_40mg_ml, acetaminophen, haloperidol
):
    assert acetaminophen.is_superclass_of(apap_500mg)
    assert haloperidol.is_superclass_of(haloperidol_40mg_ml)
    assert not haloperidol.is_superclass_of(apap_500mg)
    assert not acetaminophen.is_superclass_of(haloperidol_40mg_ml)


def test_component_match(
    haloperidol_40mg_ml,
    _haloperidol_39mg_ml,
    _haloperidol_41mg_ml,
    _haloperidol_45mg_ml,
    haloperidol_decanoate_40mg_ml,
    haloperidol_etherate_40mg_ml,
    apap_40mg_ml,
):
    # Is superclass of itself
    assert haloperidol_40mg_ml.is_superclass_of(haloperidol_40mg_ml)

    # Match window
    assert haloperidol_40mg_ml.is_superclass_of(_haloperidol_39mg_ml)
    assert not haloperidol_40mg_ml.is_superclass_of(_haloperidol_45mg_ml)
    assert not haloperidol_40mg_ml.is_superclass_of(apap_40mg_ml)

    # Generic is superclass of precise, but not vice versa
    assert haloperidol_40mg_ml.is_superclass_of(haloperidol_decanoate_40mg_ml)
    assert not haloperidol_decanoate_40mg_ml.is_superclass_of(
        haloperidol_40mg_ml
    )

    # Different PI
    assert not haloperidol_decanoate_40mg_ml.is_superclass_of(
        haloperidol_etherate_40mg_ml
    )
