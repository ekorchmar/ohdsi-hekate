import polars as pl
import pathlib

pack_content = pathlib.Path("~/Downloads/RxNE/pack_content.csv").expanduser()
pc = pl.read_csv(pack_content)

concept = pathlib.Path("~/Downloads/RxNE/CONCEPT.csv").expanduser()
c = pl.read_csv(concept, separator="\t")

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
        pl.col("concept_class_id").alias("concept_class_id_pack"),
        "concept_class_id_drug",
        "vocabulary_id",
        # NOTE: Undefined amount and box_size are implicit 1
        amount_defined=pl.col("amount").is_not_null(),
        box_size_defined=pl.col("box_size").is_not_null(),
    )
)
print(full_joined)
# full_joined.write_csv("full_joined.csv")

# Question 1: can amount and box_size be both defined and undefined?
quantity_states = (
    full_joined.group_by(
        "pack_concept_id",
        "concept_class_id_pack",
        "vocabulary_id",
    )
    .agg(
        pl.col("amount_defined").n_unique().alias("amount_states_in_pack"),
        pl.col("box_size_defined").n_unique().alias("box_size_states_in_pack"),
    )
    .group_by(
        "concept_class_id_pack",
        "vocabulary_id",
        "amount_states_in_pack",
        "box_size_states_in_pack",
    )
    .count()
)
# 3 total packs with 2 amount states
print(quantity_states)
quantity_states.write_csv("quantity_states.csv")


# Question 2: Now: which concept_class_ids have which definition states?
definition_states = (
    full_joined.filter(
        pl.col("concept_class_id_pack") != "Marketed Product",  # Fake class
    )
    .group_by(
        "concept_class_id_pack",
        "vocabulary_id",
        "amount_defined",
        "box_size_defined",
    )
    .len()
    .sort(pl.col("concept_class_id_pack").str.len_bytes())
)

print(definition_states)
definition_states.write_csv("definition_states.csv")
# *Pack Boxes can have box_size and amount, *Packs can have amount


# Question 3: What are allowed content classes?
allowed_classes = (
    full_joined.filter(
        pl.col("concept_class_id_pack") != "Marketed Product",  # Fake class
    )
    .group_by(
        "concept_class_id_pack",
        "vocabulary_id",
        "concept_class_id_drug",
    )
    .len()
    .sort(
        pl.col("concept_class_id_pack").str.len_bytes(),
        pl.col("vocabulary_id"),
        pl.col("concept_class_id_drug").str.len_bytes(),
    )
)
print(allowed_classes)
# Interesting: RxNorm allows Branded Drugs to be in a pack, RxE does not
allowed_classes.write_csv("allowed_classes.csv")
