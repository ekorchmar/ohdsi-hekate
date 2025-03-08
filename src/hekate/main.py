from pathlib import Path
import logging

from csv_read.athena import OMOPVocabulariesV5
from csv_read.source_input import BuildRxEInput


def main():
    print("""\
With wing of bat and eye of toad,
Avoiding legacy code bloat,
With twig of fern and QA checks,
Hekate starts another hex!
""")


if __name__ == "__main__":
    main()

    vocab_path = Path("~/Downloads/Vocab/").expanduser()
    source_path = Path("~/Downloads/SourceInput/").expanduser()
    athena_rxne = OMOPVocabulariesV5(vocab_download_path=vocab_path)
    logging.basicConfig(level=logging.DEBUG)
    ggr_source = BuildRxEInput(
        data_path=source_path,
        rx_atoms=athena_rxne.atoms,
    )
    for node in ggr_source.build_drug_nodes():
        print(node)

    print("Done")
