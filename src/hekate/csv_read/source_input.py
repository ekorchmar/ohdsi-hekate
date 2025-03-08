"""
Contains implementations to read TSV data from a file and transform it
into ForeignDrugNode objects for evaluation.
"""

import logging
from abc import ABC
from collections.abc import Generator
from itertools import product
from pathlib import Path
from typing import Annotated, override

import polars as pl
from csv_read.generic import CSVReader, Schema
from rx_model import drug_classes as dc
from rx_model import hierarchy as h
from utils.classes import SortedTuple
from utils.constants import STRENGTH_CONFIGURATIONS_CODE, PERCENT_CONCEPT_ID
from utils.logger import LOGGER

type _ComponentPermutations = Generator[
    dc.BoundStrength[dc.ConceptCodeVocab, dc.Strength], None, None
]


class SourceTable[IdS: pl.DataFrame | None](CSVReader[IdS], ABC):
    """
    Abstract class for reading BuildRxE input tables in CSV/TSV format.


    Attributes:
     TABLE_SCHEMA: Schema for the table.
        TABLE_COLUMNS: Ordered sequence of columns to keep from the table.
    """


class DrugConceptStage(SourceTable[None]):
    TABLE_SCHEMA: Schema = {
        "concept_code": pl.Utf8,
        "concept_name": pl.Utf8,
        "concept_class_id": pl.Utf8,
        "vocabulary_id": pl.Utf8,
        "source_concept_class_id": pl.Utf8,
        "possible_excipient": pl.Null,  # NOTE: not implemented yet
        "valid_start_date": pl.Date,
        "valid_end_date": pl.Date,
        "invalid_reason": pl.Utf8,
    }

    TABLE_COLUMNS: list[str] = [
        "concept_code",
        "concept_name",
        "concept_class_id",
        "vocabulary_id",
        "source_concept_class_id",
        # "possible_excipient",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
    ]

    @staticmethod
    def date_to_yyyymmdd(colname: str) -> pl.Expr:
        return (
            pl.col(colname).dt.year() * 1_00_00
            + pl.col(colname).dt.month() * 1_00
            + pl.col(colname).dt.day()
        )

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: None = None
    ) -> pl.LazyFrame:
        del valid_concepts
        return frame.select(
            pl.all().exclude("valid_start_date", "valid_end_date"),
            # Make dates YYYYMMDD integers
            valid_start_date=self.date_to_yyyymmdd("valid_start_date"),
            valid_end_date=self.date_to_yyyymmdd("valid_end_date"),
        )


class DSStage(SourceTable[pl.DataFrame]):
    type dss_strength_tuple = tuple[float | None, h.PseudoUnit]

    TABLE_SCHEMA: Schema = {
        "drug_concept_code": pl.Utf8,
        "ingredient_concept_code": pl.Utf8,
        "amount_value": pl.Float64,
        "amount_unit": pl.Utf8,
        "numerator_value": pl.Float64,
        "numerator_unit": pl.Utf8,
        "denominator_value": pl.Float64,
        "denominator_unit": pl.Utf8,
        "box_size": pl.UInt16,
    }

    TABLE_COLUMNS: list[str] = [
        "drug_concept_code",
        "ingredient_concept_code",
        "amount_value",
        "amount_unit",
        "numerator_value",
        "numerator_unit",
        "denominator_value",
        "denominator_unit",
        # "box_size"  # Unused for now
    ]

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: pl.DataFrame | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError("Valid concepts must be provided for DSStage.")

        return frame.filter(
            pl.col("drug_concept_code").is_in(valid_concepts["concept_code"]),
            pl.col("ingredient_concept_code").is_in(
                valid_concepts["concept_code"]
            ),
            # WARN: temporarily discard all data mentioning box_size
            pl.col("box_size").is_null(),
        ).select(pl.all().exclude("box_size"))


class RelationshipToConcept(SourceTable[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "concept_code_1": pl.Utf8,
        "vocabulary_id_1": pl.Utf8,
        "concept_id_2": pl.UInt32,
        "precedence": pl.UInt8,
        "conversion_factor": pl.Float64,
    }

    TABLE_COLUMNS: list[str] = list(TABLE_SCHEMA.keys())

    # TODO: Implement precedence. For now, discard all rows with precedence > 1
    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: pl.DataFrame | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError(
                "Valid concepts must be provided for "
                "RelationshipToConceptStage."
            )

        return frame.filter(
            pl.col("concept_code_1").is_in(valid_concepts["concept_code"]),
            pl.col("vocabulary_id_1").is_in(valid_concepts["vocabulary_id"]),
            pl.col("precedence").is_null() | (pl.col("precedence") <= 1),
        )


class InternalRelationshipStage(SourceTable[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "concept_code_1": pl.Utf8,
        "concept_code_2": pl.Utf8,
        "relationship_id": pl.UInt32,
        "precedence": pl.UInt8,
        "conversion_factor": pl.Float64,
    }

    TABLE_COLUMNS: list[str] = list(TABLE_SCHEMA.keys())

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: pl.DataFrame | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError(
                "Valid concepts must be provided for InternalRelationshipStage."
            )

        return frame.filter(
            pl.col("concept_code_1").is_in(valid_concepts["concept_code"]),
            pl.col("concept_code_2").is_in(valid_concepts["concept_code"]),
        )


class BuildRxEInput:
    """
    Class to read and prepare BuildRxE input data for evaluation.
    """

    def __init__(
        self,
        data_path: Path,
        rx_atoms: h.Atoms[dc.ConceptId],
        delimiter: str = "\t",
        quote_char: str = '"',
    ) -> None:
        self.data_path: Path = data_path

        self.logger: logging.Logger = LOGGER.getChild(self.__class__.__name__)

        # Initiate containers
        self.source_atoms: h.Atoms[dc.ConceptCodeVocab] = h.Atoms()
        self.rx_atoms: h.Atoms[dc.ConceptId] = rx_atoms
        self.atom_mapper: h.AtomMapper = h.AtomMapper(
            atom_getter=self.rx_atoms.lookup_unknown,
            unit_storage=self.rx_atoms.unit,
        )
        self.pseudo_units: list[h.PseudoUnit] = []
        self.drug_nodes: list[
            dc.DrugNode[dc.ConceptCodeVocab, dc.Strength | None]
        ] = []

        # Read and prepare data
        self.logger.info(
            f"Starting processing of BuildRxE input tables from {data_path}"
        )

        self.dcs: DrugConceptStage = DrugConceptStage(
            path=data_path / "drug_concept_stage.tsv",
            delimiter=delimiter,
            quote_char=quote_char,
        )

        # Load valid concepts and populate the storages
        vocabs = self.dcs.collect()["vocabulary_id"].unique()
        assert len(vocabs) == 1
        self.load_valid_concepts()

        self.rtcs: RelationshipToConcept = RelationshipToConcept(
            data_path / "relationship_to_concept_stage.tsv",
            reference_data=self.dcs.collect().select(
                "concept_code", "vocabulary_id"
            ),
        )

        self.atom_mapper.populate_from_frame(
            frame=self.rtcs.collect(),
            pseudo_units=self.pseudo_units,
        )

        self.ir: InternalRelationshipStage = InternalRelationshipStage(
            path=data_path / "internal_relationship_stage.tsv",
            reference_data=self.dcs.collect().select("concept_code"),
            delimiter=delimiter,
            quote_char=quote_char,
        )

        self.dss: DSStage = DSStage(
            data_path / "ds_stage.tsv",
            reference_data=self.dcs.collect().select("concept_code"),
        )

    def load_valid_concepts(self) -> None:
        """
        Load valid concepts from DrugConceptStage and populate source_atoms.

        Populates self.source_atoms with valid concepts from DrugConceptStage,
        and registers all units as pseudo-units.
        """

        atom_concepts = self.dcs.collect().select(
            "concept_code", "vocabulary_id", "concept_name", "concept_class_id"
        )

        # Units must be excluded, as they are actually pseudo-units
        self.source_atoms.add_from_frame(
            atom_concepts.filter(pl.col("concept_class_id") != "Unit")
        )

        # Register all units as pseudo-units
        self.pseudo_units += atom_concepts.filter(
            pl.col("concept_class_id") == "Unit",
        )["concept_code"].to_list()

    def build_drug_nodes(
        self,
    ) -> Generator[
        dc.ForeignDrugNode[dc.ConceptCodeVocab, dc.Strength | None], None, None
    ]:
        """
        Build DrugNodes using the DSStage and InternalRelationshipStage data.
        """
        ir = self.ir.collect()
        dcs = self.dcs.collect()

        # First, get the unique attribute data
        def get_attribute_class(class_id: str) -> pl.DataFrame:
            return dcs.filter(pl.col("concept_class_id") == class_id)

        drug_products = dcs.filter(
            pl.col("concept_class_id") == "Drug Product"
        ).select("concept_code", "vocabulary_id")
        for attr_class in ["Dosage Form", "Brand Name", "Supplier"]:
            field_name = attr_class.lower().replace(" ", "_")
            drug_products = (
                drug_products.join(
                    other=ir.join(
                        other=get_attribute_class(attr_class),
                        right_on="concept_code_2",
                        left_on="concept_code",
                        how="semi",
                    ),
                    left_on="concept_code",
                    right_on="concept_code_1",
                    how="left",
                    validate="1:1",  # TODO: Make this an external QA check
                )
                .select(
                    "concept_code",
                    "vocabulary_id",
                    attr_code=pl.col("concept_code_2"),
                    attr_vocab=pl.col("vocabulary_id"),  # Reuse the same column
                )
                .rename({
                    "attr_code": f"{field_name}_code",
                    "attr_vocab": f"{field_name}_vocab",
                })
            )

        row: Annotated[tuple[str, ...], 8]
        for row in drug_products.iter_rows():
            drug_product_id = dc.ConceptCodeVocab(row[0], row[1])
            A = self.source_atoms
            df = A.dose_form.get(dc.ConceptCodeVocab(row[2], row[3]))
            bn = A.brand_name.get(dc.ConceptCodeVocab(row[4], row[5]))
            sp = A.supplier.get(dc.ConceptCodeVocab(row[6], row[7]))

            strength_data = self.get_strength_combinations(drug_product_id)
            for strength_combination in strength_data:
                yield dc.ForeignDrugNode(
                    identifier=drug_product_id,
                    strength_data=strength_combination,
                    dose_form=df,
                    brand_name=bn,
                    supplier=sp,
                )

    def get_strength_combinations(
        self, drug_product_id: dc.ConceptCodeVocab
    ) -> Generator[
        SortedTuple[dc.BoundStrength[dc.ConceptCodeVocab, dc.Strength | None]],
        None,
        None,
    ]:
        """
        Get dc.strength combinations from the DSStage data.
        """
        strength_data = (
            self.dss.collect()
            .filter(pl.col("drug_concept_code") == drug_product_id.concept_code)
            .select(pl.all().exclude("drug_concept_code"))
        )

        # For drugs without strength data, just return the ingredients from
        # the IRS
        if len(strength_data) == 0:
            ingredient_codes = (
                self.ir.collect()
                .filter(
                    pl.col("concept_code_1") == drug_product_id.concept_code
                )
                .join(
                    self.dcs.collect(),
                    left_on="concept_code_2",
                    right_on="concept_code",
                )["concept_code", "vocabulary_id"]
            )

            strengthless: list[
                tuple[dc.Ingredient[dc.ConceptCodeVocab], None]
            ] = []
            for row in ingredient_codes.iter_rows():
                concept_code: str = row[0]
                vocab_id: str = row[1]
                ingredient = self.source_atoms.ingredient[
                    dc.ConceptCodeVocab(concept_code, vocab_id)
                ]
                strengthless.append((ingredient, None))

            # One permutation as there is only one None
            yield SortedTuple(strengthless)
            return

        # First, determine the shape of the strength data
        configuration: str = "wrong"
        for configuration, expression in STRENGTH_CONFIGURATIONS_CODE.items():
            if len(strength_data.filter(expression)) > 0:
                break

        # Define a generator for all possible permutations of individual
        # strength components
        # NOTE: This might seem like a lot of nested loops, but in practice
        # most precedence values are exactly 1, so we expect only one
        # iteration per every loop.
        component_permutations: list[_ComponentPermutations] = []
        for row in strength_data.iter_rows():
            ingredient = self.source_atoms.ingredient[
                dc.ConceptCodeVocab(row[0], row[1])
            ]

            match configuration:
                case "amount_only":
                    amount: float = row[2]
                    unit: h.PseudoUnit = row[3]
                    am_permut = self.atom_mapper.translate_strength_measure(
                        amount, unit
                    )
                    component_permutations.append(
                        (ingredient, dc.SolidStrength(scaled_v, true_unit))
                        for scaled_v, true_unit in am_permut
                    )

                case "liquid_concentration":
                    numerator: float = row[4]
                    numerator_unit: h.PseudoUnit = row[5]
                    # Denominator is implicit 1
                    denominator_unit: h.PseudoUnit = row[7]
                    num_permut = self.atom_mapper.translate_strength_measure(
                        numerator, numerator_unit
                    )
                    den_permut = self.atom_mapper.translate_strength_measure(
                        1, denominator_unit
                    )
                    component_permutations.append(
                        (
                            ingredient,
                            dc.LiquidConcentration(
                                numerator_value=scaled_n / scaled_d,
                                numerator_unit=n_unit,
                                denominator_unit=d_unit,
                            ),
                        )
                        for (scaled_n, n_unit), (scaled_d, d_unit) in product(
                            num_permut, den_permut
                        )
                    )

                case "liquid_quantity":
                    numerator: float = row[4]
                    numerator_unit: h.PseudoUnit = row[5]
                    denominator: float = row[6]
                    denominator_unit: h.PseudoUnit = row[7]
                    num_permut = self.atom_mapper.translate_strength_measure(
                        numerator, numerator_unit
                    )
                    den_permut = self.atom_mapper.translate_strength_measure(
                        denominator, denominator_unit
                    )
                    component_permutations.append(
                        (
                            ingredient,
                            dc.LiquidQuantity(
                                numerator_value=scaled_n,
                                numerator_unit=n_unit,
                                denominator_value=scaled_d,
                                denominator_unit=d_unit,
                            ),
                        )
                        for (scaled_n, n_unit), (scaled_d, d_unit) in product(
                            num_permut, den_permut
                        )
                    )
                case "gas_concentration":
                    # Those are static
                    numerator: float = row[4]
                    pct = self.rx_atoms.unit[dc.ConceptId(PERCENT_CONCEPT_ID)]
                    only_component = (
                        ingredient,
                        dc.GasPercentage(numerator, pct),
                    )
                    component_permutations.append(x for x in (only_component,))

                case "wrong":
                    raise ValueError(
                        "Strength data does not match any configuration."
                    )

        # Yield all possible permutations of strength components
        for combination in product(*component_permutations):
            yield SortedTuple(combination)
