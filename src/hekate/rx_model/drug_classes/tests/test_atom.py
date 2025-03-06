import pytest
import rx_model.drug_classes.atom as a
import rx_model.drug_classes.generic as g
from utils.classes import SortedTuple


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
def omop_haloperidol():
    return g.ConceptCodeVocab(
        "OMOP41", "RxNorm Extension"
    ), "Haloperydol Kharkivdrug"


@pytest.fixture
def omop_apap_1():
    return g.ConceptCodeVocab(
        "OMOP42", "RxNorm Extension"
    ), "Paracetamol Darnytsia"


@pytest.fixture
def omop_apap_2():
    return g.ConceptCodeVocab(
        "OMOP42", "RxNorm Extension"
    ), "Paracetamol Kharkivdrug"


@pytest.fixture
def haloperidol_decanoate(haloperidol):
    return (
        g.ConceptId(19068898),
        "haloperidol decanoate",
        a.Ingredient(*haloperidol),
    )


def test_atoms_eq(
    acetaminophen,
    paracetamol,
    tylenol,
    haloperidol,
    omop_apap_1,
    omop_apap_2,
    omop_haloperidol,
):
    # ConceptId
    assert a.Ingredient(*acetaminophen) == a.Ingredient(*paracetamol)
    assert a.Ingredient(*acetaminophen) == a.Ingredient(*tylenol)
    assert a.Ingredient(*acetaminophen) != a.Ingredient(*haloperidol)
    with pytest.raises(TypeError):
        assert a.Ingredient(*acetaminophen) != a.BrandName(*tylenol)

    # ConceptCodeVocab
    assert a.Ingredient(*omop_apap_1) == a.Ingredient(*omop_apap_2)
    assert a.Ingredient(*omop_apap_1) != a.Ingredient(*omop_haloperidol)


def test_reconstructed_invariant(
    haloperidol, haloperidol_decanoate, paracetamol
):
    ing_1 = a.Ingredient(*haloperidol)
    ing_2 = a.Ingredient(*paracetamol)

    pi = a.PreciseIngredient(*haloperidol_decanoate)

    assert ing_1 == pi.invariant
    assert ing_2 != pi.invariant


def test_node_interfaces_ingredient(
    acetaminophen, haloperidol, paracetamol, omop_apap_1
):
    acetaminophen_ing = a.Ingredient(*acetaminophen)
    haloperidol_ing = a.Ingredient(*haloperidol)
    paracetamol_ing = a.Ingredient(*paracetamol)
    omop_apap_ing = a.Ingredient(*omop_apap_1)

    ings = [
        acetaminophen_ing,
        haloperidol_ing,
        paracetamol_ing,
        omop_apap_ing,
    ]

    # Predictable attributes
    for ing in ings:
        assert ing.get_strength_data() == SortedTuple([
            (ing, None),
        ])
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


def test_node_interfaces_ingredient_superclass(
    acetaminophen, haloperidol, paracetamol, omop_apap_1
):
    acetaminophen_ing = a.Ingredient(*acetaminophen)
    haloperidol_ing = a.Ingredient(*haloperidol)
    paracetamol_ing = a.Ingredient(*paracetamol)

    # Superclass check
    assert acetaminophen_ing.is_superclass_of(paracetamol_ing)
    assert not acetaminophen_ing.is_superclass_of(haloperidol_ing)
    assert not haloperidol_ing.is_superclass_of(acetaminophen_ing)


def test_sorting_ingredient(
    acetaminophen, paracetamol, haloperidol, omop_apap_1, omop_haloperidol
):
    acetaminophen_ing = a.Ingredient(*acetaminophen)
    paracetamol_ing = a.Ingredient(*paracetamol)
    haloperidol_ing = a.Ingredient(*haloperidol)
    omop_apap_ing = a.Ingredient(*omop_apap_1)
    omop_haloperidol_ing = a.Ingredient(*omop_haloperidol)

    # Sorted by ConceptId
    for pair in (
        # ConceptId
        (acetaminophen_ing, haloperidol_ing),
        # ConceptId, equal
        (acetaminophen_ing, paracetamol_ing),
        # ConceptCodeVocab
        (omop_apap_ing, omop_haloperidol_ing),
    ):
        small, big = sorted(pair)
        assert big.identifier >= small.identifier

    # TypeError on different types
    with pytest.raises(TypeError):
        acetaminophen_ing > omop_apap_ing


if __name__ == "__main__":
    pytest.main()
