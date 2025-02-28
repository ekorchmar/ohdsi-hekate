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


if __name__ == '__main__':
    main()
    path = Path("~/Downloads/Vocab/").expanduser()
    LOGGER.info(f"Starting processing of Athena Vocabularies from {path}")
    a = OMOPVocabulariesV5(vocab_download_path=path)
    counts = {k: len(v) for k, v in a.atoms.precise_ingredient.items()}
    from collections import Counter
    import json

    c = Counter(counts.values())
    print(json.dumps(c, indent=2))
    print("Done")
