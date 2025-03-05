from csv_read.athena import OMOPVocabulariesV5
from pathlib import Path
from utils.logger import LOGGER


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
    LOGGER.info(f"Starting processing of Athena Vocabularies from {path}")
    a = OMOPVocabulariesV5(vocab_download_path=path)

    import rustworkx as rx
    from rx_model.drug_classes import ConceptId

    from rustworkx.visualization import graphviz_draw

    aspirin = a.atoms.ingredient[ConceptId(1112807)]
    aspirin_node = a.hierarchy.ingredients[aspirin]
    aspirin_descendants = rx.descendants(a.hierarchy.graph, aspirin_node)
    aspirin_subgraph = a.hierarchy.graph.subgraph(list(aspirin_descendants))
    print(aspirin_subgraph.num_edges())  # Expect 9431

    img = graphviz_draw(aspirin_subgraph)
    assert img is not None
    img.show()

    print("Done")
