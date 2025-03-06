import pytest
import rx_model.drug_classes.atom as a
import rx_model.drug_classes.generic as g
from utils.classes import SortedTuple


# All fixtures exist for all of the test functions
permafixture = pytest.fixture(scope="package")


# ConceptId
## Ingredients
@permafixture
def acetaminophen():
    return a.Ingredient(g.ConceptId(1125315), "acetaminophen")


@permafixture
def paracetamol():
    return a.Ingredient(g.ConceptId(1125315), "paracetamol")


@permafixture
def haloperidol():
    return a.Ingredient(g.ConceptId(766529), "haloperidol")


## Precise Ingredients
@permafixture
def haloperidol_decanoate(haloperidol):
    return a.PreciseIngredient(
        g.ConceptId(19068898),
        "haloperidol decanoate",
        haloperidol,
    )


## Brand Names
@permafixture
def _tylenol():
    return a.BrandName(g.ConceptId(1125315), "Tylenol")


@permafixture
def advicor():
    return a.BrandName(g.ConceptId(19082896), "Advicor")


@permafixture
def bupap():
    return a.BrandName(g.ConceptId(19057227), "Bupap")


@permafixture
def oral_tablet():
    return a.DoseForm(g.ConceptId(19082573), "Oral Tablet")


@permafixture
def oral_solution():
    return a.DoseForm(g.ConceptId(19082170), "Oral Solution")


# Unit
@permafixture
def mg():
    return a.Unit(g.ConceptId(8576), "mg")


@permafixture
def ml():
    return a.Unit(g.ConceptId(8587), "ml")


# ConceptCodeVocab
## Ingredients
@permafixture
def omop_haloperidol():
    return a.Ingredient(
        g.ConceptCodeVocab("OMOP41", "RxNorm Extension"),
        "Haloperydol Kharkivdrug",
    )


@permafixture
def omop_apap_1():
    return a.Ingredient(
        g.ConceptCodeVocab("OMOP42", "RxNorm Extension"),
        "Paracetamol Darnytsia",
    )


@permafixture
def omop_apap_2():
    return a.Ingredient(
        g.ConceptCodeVocab("OMOP42", "RxNorm Extension"),
        "Paracetamol Kharkivdrug",
    )


def test_atoms_eq(
    acetaminophen,
    paracetamol,
    _tylenol,
    haloperidol,
    omop_apap_1,
    omop_apap_2,
    omop_haloperidol,
):
    # ConceptId
    assert acetaminophen == paracetamol
    assert acetaminophen != haloperidol

    # TypeError on different types
    with pytest.raises(TypeError):
        assert acetaminophen != _tylenol
    # But not on None
    assert acetaminophen != None  # noqa: E711

    # ConceptCodeVocab
    assert omop_apap_1 == omop_apap_2
    assert omop_apap_1 != omop_haloperidol


def test_reconstructed_invariant(
    haloperidol, haloperidol_decanoate, paracetamol
):
    ing_1 = haloperidol
    ing_2 = paracetamol

    pi = haloperidol_decanoate

    assert ing_1 == pi.invariant
    assert ing_2 != pi.invariant


def test_node_interfaces_ingredient(
    acetaminophen, haloperidol, paracetamol, omop_apap_1
):
    acetaminophen_ing = acetaminophen
    haloperidol_ing = haloperidol
    paracetamol_ing = paracetamol
    omop_apap_ing = omop_apap_1

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
    acetaminophen_ing = acetaminophen
    haloperidol_ing = haloperidol
    paracetamol_ing = paracetamol

    # Superclass check
    assert acetaminophen_ing.is_superclass_of(paracetamol_ing)
    assert not acetaminophen_ing.is_superclass_of(haloperidol_ing)
    assert not haloperidol_ing.is_superclass_of(acetaminophen_ing)


def test_sorting_ingredient(
    acetaminophen, paracetamol, haloperidol, omop_apap_1, omop_haloperidol
):
    # Sorted by ConceptId
    for pair in (
        # ConceptId
        (acetaminophen, haloperidol),
        # ConceptId, equal
        (acetaminophen, paracetamol),
        # ConceptCodeVocab
        (omop_apap_1, omop_haloperidol),
    ):
        small, big = sorted(pair)
        assert big.identifier >= small.identifier

    # TypeError on different types
    with pytest.raises(TypeError):
        acetaminophen > omop_apap_1


if __name__ == "__main__":
    pytest.main()
