from pathlib import Path

import polars as pl
import rx_model.drug_classes as dc
from csv_read.athena import OMOPVocabulariesV5
from rustworkx.visualization import graphviz_draw
from rx_model.drug_classes import DRUG_CLASS_PREFERENCE_ORDER
from rx_model.hierarchy.traversal import DrugNodeFinder
from utils.classes import SortedTuple


def main():
    print("""\
With wing of bat and eye of toad,
Avoiding legacy code bloat,
With twig of fern and QA checks,
Hekate starts another hex!
""")


if __name__ == "__main__":
    main()

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
    matched_nodes = finder.get_search_results()

    class_counts = {class_: 0 for class_ in DRUG_CLASS_PREFERENCE_ORDER}

    for node in matched_nodes.values():
        for class_ in DRUG_CLASS_PREFERENCE_ORDER:
            if isinstance(node, class_):
                class_counts[class_] += 1
                break
    for class_, count in class_counts.items():
        print(f"{class_.__name__}: {count}")

    matched_tree = athena_rxne.hierarchy.graph.subgraph(list(matched_nodes))
    img = graphviz_draw(
        matched_tree,
        node_attr_fn=lambda drug_node: {"label": str(drug_node.identifier)},
    )

    assert img is not None
    img.save("matched_tree.png")
    img.show()

    concepts = athena_rxne.concept.data().filter(
        pl.col("concept_id").is_in(
            pl.Series(
                (node.identifier for node in matched_nodes.values()),
                dtype=pl.UInt32,
            )
        )
    )
    concepts.write_excel("matched_concepts.xlsx")

    print("Done")
