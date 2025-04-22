import polars as pl
import pathlib

pack_content = pathlib.Path("~/Downloads/RxNE/pack_content.csv").expanduser()
pc = pl.read_csv(pack_content)

concept = pathlib.Path("~/Downloads/RxNE/CONCEPT.csv").expanduser()
c = pl.read_csv(concept, separator="\t")

concept_relationship = pathlib.Path(
    "~/Downloads/RxNE/CONCEPT_RELATIONSHIP.csv"
).expanduser()
cr = pl.read_csv(concept_relationship, separator="\t")

drug_strength = pathlib.Path("~/Downloads/RxNE/DRUG_STRENGTH.csv").expanduser()
ds = pl.read_csv(drug_strength, separator="\t")


full_joined = (
    # Pack class
    pc.join(
        other=c,
        how="inner",
        left_on="pack_concept_id",
        right_on="concept_id",
        suffix="_pack",
    )
    # Drug class
    .join(
        other=c,
        how="inner",
        left_on="drug_concept_id",
        right_on="concept_id",
        suffix="_drug",
    )
    .select(
        "pack_concept_id",
        "drug_concept_id",
        "amount",
        "box_size",
        pl.col("concept_class_id").alias("concept_class_id_pack"),
        pl.col("vocabulary_id").alias("vocabulary_id_pack"),
        "concept_class_id_drug",
        "vocabulary_id_drug",
    )
)
print(full_joined)
# full_joined.write_csv("full_joined.csv")

# Question 1: are there monocomponent packs in RxNorm (or RxE) that are modeled
# as box_size drugs in RxE?
rxnorm_mono_pack = full_joined.join(
    other=(
        full_joined.group_by("pack_concept_id")
        .len("components")
        .filter(pl.col("components") == 1)
    ),
    on="pack_concept_id",
    how="semi",
).filter(
    # NOTE: Limit to clinical packs
    pl.col("concept_class_id_pack") == "Clinical Pack"
)
# print(rxnorm_mono_pack)

rxe_box_sized = (
    c.filter(
        pl.col("concept_class_id") == "Clinical Drug Box",
        pl.col("vocabulary_id") == "RxNorm Extension",
        pl.col("standard_concept") == "S",
    )
    .join(
        other=cr.filter(
            pl.col("invalid_reason").is_null(),
            pl.col("relationship_id") == "Box of",  # (Quant) Clinical Durg
        ),
        left_on="concept_id",
        right_on="concept_id_1",
        suffix="_CR",
    )
    .join(
        other=ds.unique(["drug_concept_id", "box_size"]),
        left_on="concept_id",
        right_on="drug_concept_id",
        suffix="_DS",
    )
    .select(
        drug_concept_id=pl.col("concept_id"),
        content_concept_id=pl.col("concept_id_2"),
        box_size=pl.col("box_size"),
    )
)
# print(rxe_box_sized)

modelling_intersection = rxnorm_mono_pack.join(
    other=rxe_box_sized,
    left_on=["drug_concept_id", "amount"],
    right_on=["content_concept_id", "box_size"],
).rename({"drug_concept_id_right": "rxe_cdb_concept_id"})
print(modelling_intersection)
modelling_intersection.write_csv("modelling_intersection.csv")

# Question 2: Does separating box_size from amount bring semantic duplicates?
rxe_cpb = (
    full_joined.filter(
        pl.col("vocabulary_id_pack") == "RxNorm Extension",
        pl.col("concept_class_id_pack") == "Clinical Pack Box",
    )
    .select(
        cpb_concept_id="pack_concept_id",
        drug_concept_id="drug_concept_id",
        amount=pl.coalesce(pl.col("amount"), 1) * pl.col("box_size"),
    )
    .sort("drug_concept_id")
    .group_by("cpb_concept_id")
    .all()
)

rxn_cp = (
    full_joined.filter(
        pl.col("vocabulary_id_pack") == "RxNorm",
        pl.col("concept_class_id_pack") == "Clinical Pack",
    )
    .select(
        "pack_concept_id",
        "drug_concept_id",
        "amount",
    )
    .sort("drug_concept_id")
    .group_by("pack_concept_id")
    .all()
)

interlevel_intersection = (
    rxn_cp.join(other=rxe_cpb, on=["drug_concept_id", "amount"])
    .select("pack_concept_id", "cpb_concept_id", "drug_concept_id", "amount")
    .explode("drug_concept_id", "amount")
)
print(interlevel_intersection)
interlevel_intersection.write_csv("interlevel_intersection.csv")
