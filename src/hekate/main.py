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
    import matplotlib.pyplot as plt
    from rustworkx.visualization import mpl_draw
    from rx_model.drug_classes import ConceptId

    aspirin = a.atoms.ingredient[ConceptId(1112807)]
    aspirin_node = a.hierarchy.ingredients[aspirin]
    aspirin_descendants = rx.descendants(a.hierarchy.graph, aspirin_node)
    aspirin_subgraph = a.hierarchy.graph.subgraph(list(aspirin_descendants))
    _ = mpl_draw(aspirin_subgraph, with_labels=True)
    plt.show()

    print("Done")
