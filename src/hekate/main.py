from typing import override


def _main():
    # runner = HekateRunner()
    # runner.run()
    # runner.write_results()

    import datetime
    import os
    import pathlib
    from collections import defaultdict

    import polars as pl
    import rustworkx as rx
    from csv_read.athena import OMOPVocabulariesV5  # for Athena data
    from rx_model import drug_classes as dc

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
            "INGREDIENT",
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

    # Writing down complex concepts
    cdc_schema = {
        **base_df.schema,
        "ingredient_id": pl.Int64,
        "amount": pl.Float64,
        "amount_unit": pl.String,
        "denominator_unit": pl.String,
    }
    cdc = []

    cdf_schema = {
        **base_df.schema,
        "dose_form_id": pl.Int64,
    }
    cdf = []

    cdf_ingredient_schema = {
        "cdf_id": pl.Int64,
        "ingredient_id": pl.Int64,
    }
    cdf_ingredients = []

    # A table for clinical drugs and BDC to store unique CDC combinations
    # Use dict as ordered set
    drug_content = {}
    drug_content_component_schema = {
        "content_id": pl.Int64,
        "cdc_id": pl.Int64,
    }
    dcc = []

    clinical_drug_schema = {
        **base_df.schema,
        "cdf_id": pl.Int64,
        "content_id": pl.Int64,
    }
    cd = []

    branded_drug_component_schema = {
        **base_df.schema,
        "brand_name_id": pl.Int64,
        "content_id": pl.Int64,
    }
    bdc = []

    branded_drug_form_schema = {
        **base_df.schema,
        "cdf_id": pl.Int64,
        "brand_name_id": pl.Int64,
    }
    bdf = []

    branded_drug_schema = {
        **base_df.schema,
        "cd_id": pl.Int64,
        "bdc_id": pl.Int64,
        "bdf_id": pl.Int64,
    }
    bd = []

    quantified_clinical_drug_schema = {
        **base_df.schema,
        "cd_id": pl.Int64,
        "denominator": pl.Float64,
    }
    qcd = []

    quantified_branded_drug_schema = {
        **base_df.schema,
        "qcd_id": pl.Int64,
        "brand_name_id": pl.Int64,
    }
    qbd = []

    names: dict[int, str] = defaultdict(
        (lambda: "Unknown"),
        {
            row["concept_id"]: row["concept_name"]
            for row in athena.concept.collect().iter_rows(named=True)
        },
    )

    def get_drug_content_id(
        node: dc.ClinicalDrug | dc.BrandedDrugComponent,
    ) -> int:
        drug_content_id = drug_content.get(node.clinical_drug_components)

        if drug_content_id is None:
            drug_content_id = len(drug_content) + 1
            drug_content[node.clinical_drug_components] = drug_content_id

            for cdc_node in node.clinical_drug_components:
                cdc_id = cdc_node.identifier
                dcc.append({
                    "content_id": drug_content_id,
                    "cdc_id": cdc_id,
                })

        return drug_content_id

    # Define a crawler to traverse the hierarrchy of concepts
    class Crawler(rx.visit.BFSVisitor):
        @override
        def discover_vertex(self, v: int):
            node = athena.hierarchy[v]
            match node:
                case dc.ClinicalDrugComponent():
                    match node.strength:
                        case dc.SolidStrength():
                            cdc.append({
                                "concept_id": node.identifier,
                                "concept_name": names[node.identifier],
                                "ingredient_id": node.ingredient.identifier,
                                "amount": node.strength.amount_value,
                                "amount_unit": node.strength.amount_unit.concept_name,
                                "denominator_unit": None,
                            })
                        case dc.LiquidConcentration():
                            cdc.append({
                                "concept_id": node.identifier,
                                "concept_name": names[node.identifier],
                                "ingredient_id": node.ingredient.identifier,
                                "amount": node.strength.numerator_value,
                                "amount_unit": node.strength.numerator_unit.concept_name,
                                "denominator_unit": node.strength.denominator_unit.concept_name,
                            })
                        case dc.GasPercentage():
                            cdc.append({
                                "concept_id": node.identifier,
                                "concept_name": names[node.identifier],
                                "ingredient_id": node.ingredient.identifier,
                                "amount": node.strength.numerator_value,
                                "amount_unit": "%",
                                "denominator_unit": None,
                            })
                        case _:
                            raise TypeError(
                                f"Unexpected strength type: {node.strength}"
                            )
                case dc.ClinicalDrugForm():
                    cdf.append({
                        "concept_id": node.identifier,
                        "concept_name": names[node.identifier],
                        "dose_form_id": node.dose_form.identifier,
                    })
                    cdf_ingredients.extend(
                        {
                            "cdf_id": node.identifier,
                            "ingredient_id": ingredient.identifier,
                        }
                        for ingredient in node.ingredients
                    )

                case dc.ClinicalDrug():
                    content_id = get_drug_content_id(node)
                    cd.append({
                        "concept_id": node.identifier,
                        "concept_name": names[node.identifier],
                        "cdf_id": node.form.identifier,
                        "content_id": content_id,
                    })

                case dc.BrandedDrugComponent():
                    content_id = get_drug_content_id(node)
                    bdc.append({
                        "concept_id": node.identifier,
                        "concept_name": names[node.identifier],
                        "brand_name_id": node.brand_name.identifier,
                        "content_id": content_id,
                    })

                case dc.BrandedDrugForm():
                    bdf.append({
                        "concept_id": node.identifier,
                        "concept_name": names[node.identifier],
                        "cdf_id": node.clinical_drug_form.identifier,
                        "brand_name_id": node.brand_name.identifier,
                    })

                case dc.BrandedDrug():
                    bd.append({
                        "concept_id": node.identifier,
                        "concept_name": names[node.identifier],
                        "cd_id": node.clinical_drug.identifier,
                        "bdc_id": node.branded_component.identifier,
                        "bdf_id": node.branded_form.identifier,
                    })

                case dc.QuantifiedClinicalDrug():
                    qcd.append({
                        "concept_id": node.identifier,
                        "concept_name": names[node.identifier],
                        "cd_id": node.unquantified.identifier,
                        "denominator": node.contents[0][1].denominator_value,
                    })

                case dc.QuantifiedBrandedDrug():
                    qbd.append({
                        "concept_id": node.identifier,
                        "concept_name": names[node.identifier],
                        "qcd_id": node.unbranded.identifier,
                        "brand_name_id": node.brand_name.identifier,
                    })

                case _:
                    # Unsupported node type, skip it
                    return

    crawler = Crawler()
    rx.bfs_search(athena.hierarchy, None, crawler)

    for name, data, schema in (
        ("CLINICAL_DRUG_COMPONENT", cdc, cdc_schema),
        ("CLINICAL_DRUG_FORM", cdf, cdf_schema),
        ("CDF_INGREDIENT", cdf_ingredients, cdf_ingredient_schema),
        ("DRUG_CONTENT_COMPONENT", dcc, drug_content_component_schema),
        ("CLINICAL_DRUG", cd, clinical_drug_schema),
        ("BRANDED_DRUG_COMPONENT", bdc, branded_drug_component_schema),
        ("BRANDED_DRUG_FORM", bdf, branded_drug_form_schema),
        ("BRANDED_DRUG", bd, branded_drug_schema),
        ("QUANTIFIED_CLINICAL_DRUG", qcd, quantified_clinical_drug_schema),
        ("QUANTIFIED_BRANDED_DRUG", qbd, quantified_branded_drug_schema),
    ):
        pl.DataFrame(data, schema=schema).write_csv(out_dir / f"{name}.csv")

    pl.DataFrame(
        data=list(drug_content.values()),
        schema={"content_id": pl.Int64},
    ).write_csv(out_dir / "DRUG_CONTENT.csv")


if __name__ == "__main__":
    import sys

    sys.argv = (
        "hekate -a $ATHENA_DOWNLOAD_DIR -b $BUILD_RXE_DOWNLOAD_DIR".split()
    )
    _main()
