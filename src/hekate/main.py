from csv_read.athena import OMOPVocabulariesV5
from pathlib import Path
import rx_model.drug_classes as dc
from utils.classes import SortedTuple
from rx_model.hierarchy.traversal import DrugNodeFinder


def main():
    print("""\
With wing of bat and eye of toad,
Avoiding legacy code bloat,
With twig of fern and QA checks,
Hekate starts another hex!
""")


if __name__ == "__main__":
    main()

    import inspect

    print(inspect.signature(dc.PreciseIngredient.__init__))
    print(dc.PreciseIngredient.__init__)

    path = Path("~/Downloads/Vocab/").expanduser()
    athena_rxne = OMOPVocabulariesV5(vocab_download_path=path)

    apap = dc.ForeignDrugNode(
        identifier=dc.ConceptId(0),
        strength_data=SortedTuple([
            (
                dc.Ingredient(dc.ConceptId(1125315), "acetaminophen"),
                dc.SolidStrength(500, dc.Unit(dc.ConceptId(8576), "mg")),
            ),
        ]),
        dose_form=dc.DoseForm(dc.ConceptId(19082573), "Oral Tablet"),
    )

    finder = DrugNodeFinder(apap)

    finder.start_search(athena_rxne.hierarchy)
    print(finder.get_search_results())

    print("Done")
