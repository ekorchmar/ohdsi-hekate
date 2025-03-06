import pytest

import rx_model.drug_classes.generic as g
import rx_model.drug_classes.atom as a


@pytest.fixture
def acetaminophen():
    return g.ConceptId(1125315), "acetaminophen"


@pytest.fixture
def tylenol():
    return g.ConceptId(1125315), "Tylenol"


@pytest.fixture
def paracetamol():
    return g.ConceptId(1125315), "paracetamol"


@pytest.fixture
def haloperidol():
    return g.ConceptId(766529), "haloperidol"


@pytest.fixture
def haloperidol_decanoate():
    return g.ConceptId(19068898), "haloperidol decanoate", haloperidol


def test_atoms_eq(acetaminophen, paracetamol, tylenol, haloperidol):
    assert a.Ingredient(*acetaminophen) == a.Ingredient(*paracetamol)
    assert a.Ingredient(*acetaminophen) == a.Ingredient(*tylenol)
    assert a.Ingredient(*acetaminophen) != a.Ingredient(*haloperidol)
    assert a.Ingredient(*acetaminophen) != a.BrandName(*tylenol)


def test_reconstructed_invariant(
    haloperidol, haloperidol_decanoate, paracetamol
):
    assert (
        a.Ingredient(*haloperidol)
        == a.PreciseIngredient(*haloperidol_decanoate).invariant
    )
    assert (
        a.Ingredient(*paracetamol)
        != a.PreciseIngredient(*haloperidol_decanoate).invariant
    )


def test_node_interfaces_ingredient(acetaminophen, haloperidol, paracetamol):
    acetaminophen = a.Ingredient(*acetaminophen)
    haloperidol = a.Ingredient(*haloperidol)

    ings = [acetaminophen, haloperidol, paracetamol]

    # Predictable attributes
    for ing in ings:
        assert ing.get_strength_data() == [(ing, None)]
        assert ing.get_precise_ingredients() == [None]
        assert all(
            attr is None
            for attr in [
                ing.get_dose_form(),
                ing.get_supplier(),
                ing.get_brand_name(),
            ]
        )

        # Superclass check
        assert ing.is_superclass_of(ing)

    # Superclass check
    assert not acetaminophen.is_superclass_of(haloperidol)
    assert not haloperidol.is_superclass_of(acetaminophen)
    assert acetaminophen.is_superclass_of(paracetamol)
