def _main():
    # runner = HekateRunner()
    # runner.run()
    # runner.write_results()

    import os
    import pathlib
    import polars as pl
    from csv_read.athena import OMOPVocabulariesV5  # for Athena data
    import datetime

    vocab_path = pathlib.Path(str(os.getenv("ATHENA_DOWNLOAD_DIR")))
    athena = OMOPVocabulariesV5(vocab_download_path=vocab_path)
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = pathlib.Path.cwd() / "run" / time_now
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = pl.DataFrame(
        schema={
            "concept_id": pl.Int64,
            "concept_name": pl.String,
        }
    )
    # Writing down atoms
    atoms = [
        (
            "BRAND_NAME",
            athena.atoms.brand_name,
        ),
        (
            "INGREDIENT_NAME",
            athena.atoms.ingredient,
        ),
        (
            "DOSE_FORM",
            athena.atoms.dose_form,
        ),
    ]
    for atom_name, atom_container in atoms:
        atom_df = pl.DataFrame(
            [
                {
                    "concept_id": concept_id,
                    "concept_name": atom_object.concept_name,
                }
                for concept_id, atom_object in atom_container.items()
            ],
            schema=base_df.schema,
        )
        atom_df.write_csv(out_dir / f"{atom_name}.csv")


if __name__ == "__main__":
    import sys

    sys.argv = (
        "hekate -a $ATHENA_DOWNLOAD_DIR -b $BUILD_RXE_DOWNLOAD_DIR".split()
    )
    _main()
