import polars as pl
import pathlib

pack_content = pathlib.Path("~/Downloads/RxNE/pack_content.csv").expanduser()
pc = pl.read_csv(pack_content)

concept = pathlib.Path("~/Downloads/RxNE/CONCEPT.csv").expanduser()
c = pl.read_csv(concept, separator="\t").filter(
    pl.col("standard_concept") == "S",
)

relationship = pathlib.Path(
    "~/Downloads/RxNE/CONCEPT_RELATIONSHIP.csv"
).expanduser()
r = pl.read_csv(relationship, separator="\t").filter(
    pl.col("relationship_id") == "Marketed form of",
    pl.col("invalid_reason").is_null(),
)

ancestor = pathlib.Path("~/Downloads/RxNE/CONCEPT_ANCESTOR.csv").expanduser()
a = (
    pl.read_csv(ancestor, separator="\t")
    .filter(pl.col("ancestor_concept_id") != pl.col("descendant_concept_id"))
    .join(
        other=c,
        left_on="ancestor_concept_id",
        right_on="concept_id",
        how="semi",
    )
    .join(
        other=c,
        left_on="descendant_concept_id",
        right_on="concept_id",
        how="semi",
    )
)


marketed_product_to_parent_dirty = (
    r.join(
        other=c.filter(pl.col("concept_class_id") == "Marketed Product"),
        right_on="concept_id",
        left_on="concept_id_1",
        how="semi",
    )
    .join(
        other=c,
        right_on="concept_id",
        left_on="concept_id_2",
        how="inner",
    )
    .select("concept_id_1", "concept_id_2", "concept_class_id")
)


# Count of target per marketed product
def print_descendant_count(dataframe: pl.DataFrame) -> None:
    print(
        dataframe.group_by("concept_id_1")
        .count()
        .rename({"count": "descendant_count"})
        .group_by("descendant_count")
        .count()
        .sort("descendant_count", descending=True)
    )


print_descendant_count(marketed_product_to_parent_dirty)

ancestors = (
    a.select("ancestor_concept_id", "descendant_concept_id")
    .group_by("descendant_concept_id")
    .all()
)
marketed_product_to_redundant_ancestors = (
    marketed_product_to_parent_dirty.join(
        other=ancestors,
        left_on="concept_id_2",
        right_on="descendant_concept_id",
        how="inner",
    )
    .select("concept_id_1", "ancestor_concept_id")
    .explode("ancestor_concept_id")
    .unique(["concept_id_1", "ancestor_concept_id"])
)

# Now, remove ancestors
marketed_product_to_parent = marketed_product_to_parent_dirty.join(
    other=marketed_product_to_redundant_ancestors,
    left_on=["concept_id_1", "concept_id_2"],
    right_on=["concept_id_1", "ancestor_concept_id"],
    how="anti",
).select("concept_id_1", "concept_id_2", "concept_class_id")

print_descendant_count(marketed_product_to_parent)
# 3144 have 2 immediate ancestors, and 44 have 3. So cleanup is needed.
