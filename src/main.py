from csv_read.athena import OMOPVocabulariesV5
from pathlib import Path


def main():
    print("""\
With tail of newt and eye of toad,
Avoiding legacy code bloat,
With leaf of fern and QA checks,
Hekate starts another hex!
""")


if __name__ == '__main__':
    main()
    path = Path("~/Downloads/Vocab/").expanduser()
    a = OMOPVocabulariesV5(vocab_download_path=path)
    for ing, precise in a.atoms.precise_ingredient.items():
        print(ing, precise)
    print("Done")
