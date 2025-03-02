from csv_read.athena import OMOPVocabulariesV5
from pathlib import Path
from utils.logger import LOGGER


def main():
    print("""\
With tail of newt and wing of toad,
Avoiding legacy code bloat,
With twig of fern and QA checks,
Hekate starts another hex!
""")


if __name__ == "__main__":
    main()
    path = Path("~/Downloads/Vocab/").expanduser()
    LOGGER.info(f"Starting processing of Athena Vocabularies from {path}")
    a = OMOPVocabulariesV5(vocab_download_path=path)

    from typing import override
    import rustworkx as rx
    from rx_model.drug_classes import ConceptId, ClinicalDrugForm

    # BUG: 43147050 has 8 valid "Ingredient" relations, one of which is
    # actually a "Precise Ingredient", and another is invalid.
    # Current behavior is to "accept" it as a CDC with 6(not 8) ingredients,
    # but we may want to raise an error instead.

    class IFind43147050(rx.visit.DFSVisitor):
        @override
        def discover_vertex(self, v: int, t: int):
            del t  # Discover time
            node: ClinicalDrugForm[ConceptId] = a.hierarchy.graph[v]
            if node.identifier == ConceptId(43147050):
                for ing in node.ingredients:
                    print(ing)
                raise ValueError("Found it!")

    rx.dfs_search(
        a.hierarchy.graph,
        list(a.hierarchy.ingredients.values()),
        IFind43147050(),
    )

    print("Done")
