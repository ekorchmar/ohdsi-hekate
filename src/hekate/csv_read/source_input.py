"""
Contains implementations to read TSV data from a file and transform it
into ForeignDrugNode objects for evaluation.
"""

import logging
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Annotated, override

import polars as pl
from csv_read.generic import CSVReader, Schema
from rx_model import drug_classes as dc
from rx_model import hierarchy as h
from utils.exceptions import ForeignNodeCreationError
from utils.classes import RealNumber
from utils.logger import LOGGER


class DrugConceptStage(CSVReader[None]):
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


class DSStage(CSVReader[pl.DataFrame]):
    type dss_strength_tuple = tuple[float | None, dc.PseudoUnit]

    TABLE_SCHEMA: Schema = {
        "drug_concept_code": pl.Utf8,
        "ingredient_concept_code": pl.Utf8,
        "amount_value": RealNumber,
        "amount_unit": pl.Utf8,
        "numerator_value": RealNumber,
        "numerator_unit": pl.Utf8,
        "denominator_value": RealNumber,
        "denominator_unit": pl.Utf8,
        "box_size": pl.UInt16,
    }

    TABLE_COLUMNS: list[str] = [
        "drug_concept_code",
        "ingredient_concept_code",
        # Use matching order for strength tuple
        *dc.ForeignStrength._fields,
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
        )


class RelationshipToConcept(CSVReader[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "concept_code_1": pl.Utf8,
        "vocabulary_id_1": pl.Utf8,
        "concept_id_2": pl.UInt32,
        "precedence": pl.UInt8,
        "conversion_factor": RealNumber,
    }

    TABLE_COLUMNS: list[str] = list(TABLE_SCHEMA.keys())

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


class InternalRelationshipStage(CSVReader[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "concept_code_1": pl.Utf8,
        "concept_code_2": pl.Utf8,
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


class PCSStage(CSVReader[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "pack_concept_code": pl.Utf8,
        "drug_concept_code": pl.Utf8,
        "amount": pl.UInt16,
        "box_size": pl.UInt16,
    }
    TABLE_COLUMNS: list[str] = list(TABLE_SCHEMA.keys())

    @override
    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: pl.DataFrame | None = None
    ) -> pl.LazyFrame:
        if valid_concepts is None:
            raise ValueError("Valid concepts must be provided for PCSStage.")

        return frame.filter(
            pl.col("pack_concept_code").is_in(valid_concepts["concept_code"]),
            pl.col("drug_concept_code").is_in(valid_concepts["concept_code"]),
        )


class BuildRxEInput:
    """
    Class to read and prepare BuildRxE input data for evaluation.
    """

    def __init__(
        self,
        data_path: Path,
        delimiter: str = "\t",
        quote_char: str | None = None,
    ) -> None:
        self.data_path: Path = data_path

        self.logger: logging.Logger = LOGGER.getChild(self.__class__.__name__)

        # Initiate containers
        self.source_atoms: h.Atoms[dc.ConceptCodeVocab] = h.Atoms(self.logger)
        self.pseudo_units: list[dc.PseudoUnit] = []
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
            data_path / "relationship_to_concept.tsv",
            reference_data=self.dcs.collect().select(
                "concept_code", "vocabulary_id"
            ),
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

        # WARN: temporarily cleaning up all concepts with box_size
        boxed_drugs = self.dss.collect().filter(
            pl.col("box_size").is_not_null()
        )
        if len(boxed_drugs) > 0:
            self.logger.warning(
                f"Found {len(boxed_drugs)} drugs with box_size. "
                "These will be ignored for now."
            )
        self.dcs.anti_join(
            boxed_drugs,
            left_on="concept_code",
            right_on="drug_concept_code",
        )

        # WARN: temporarily cleaning up all pack_concepts
        self.pcs: PCSStage = PCSStage(
            data_path / "pc_stage.tsv",
            reference_data=self.dcs.collect().select("concept_code"),
        )
        if len(self.pcs.collect()) > 0:
            self.logger.warning(
                f"Found {len(self.pcs.collect())} pack_concepts. "
                "These will be ignored for now."
            )
        self.dcs.anti_join(
            self.pcs.collect(),
            left_on="concept_code",
            right_on="pack_concept_code",
        )

    def load_valid_concepts(self) -> None:
        """
        Load valid concepts from DrugConceptStage and populate source_atoms.

        Populates self.source_atoms with valid concepts from DrugConceptStage,
        and registers all units as pseudo-units.
        """

        atom_concepts = (
            self.dcs.collect()
            .filter(
                pl.col("concept_class_id").is_in([
                    "Ingredient",
                    "Dose Form",
                    "Brand Name",
                    "Supplier",
                    "Unit",
                ])
            )
            .select(
                "concept_code",
                "vocabulary_id",
                "concept_name",
                "concept_class_id",
            )
        )

        self.logger.info(f"Loaded {len(atom_concepts)} valid concepts.")

        # Units must be excluded, as they are actually pseudo-units
        self.source_atoms.add_from_frame(
            atom_concepts.filter(pl.col("concept_class_id") != "Unit")
        )

        # Register all units as pseudo-units
        self.pseudo_units += atom_concepts.filter(
            pl.col("concept_class_id") == "Unit",
        )["concept_code"].to_list()

    def prepare_drug_nodes(
        self, crash_on_error: bool = False
    ) -> Generator[dc.ForeignNodePrototype, None, None]:
        """
        Build Node prototypes using the DSStage and InternalRelationshipStage
        data.
        """
        ir = self.ir.collect()
        dcs = self.dcs.collect()

        # First, get the unique attribute data
        drug_products = (
            dcs.filter(pl.col("concept_class_id") == "Drug Product")
            .join(
                # Exclude Drug Products having explicit mappings in RTC
                other=self.rtcs.collect(),
                left_on=["concept_code", "vocabulary_id"],
                right_on=["concept_code_1", "vocabulary_id_1"],
                how="anti",
            )
            .select("concept_code", "vocabulary_id")
        )

        for attr_class in ["Dose Form", "Brand Name", "Supplier"]:
            ir_of_attr = ir.join(
                other=dcs.filter(pl.col("concept_class_id") == attr_class),
                left_on="concept_code_2",
                right_on="concept_code",
                how="semi",
            )

            field_name = attr_class.lower().replace(" ", "_")
            try:
                drug_products = (
                    drug_products.join(
                        other=ir_of_attr,
                        left_on="concept_code",
                        right_on="concept_code_1",
                        how="left",
                        validate="1:1",  # TODO: Make this an external QA check
                    )
                    .with_columns(**{
                        f"{field_name}_code": pl.col("concept_code_2"),
                    })
                    .drop("concept_code_2")
                )
            except pl.exceptions.ComputeError as e:
                if crash_on_error:
                    raise e
                self.logger.error(
                    f"Error while validating uniqueness of {attr_class} data "
                    f"for Drug Products: {e}"
                )

        row: Annotated[tuple[str, ...], 5]
        for row in drug_products.iter_rows():
            vocab = row[1]
            drug_product_id = dc.ConceptCodeVocab(row[0], vocab)
            A = self.source_atoms

            try:
                node_strength = self.get_concept_strength(drug_product_id)
            except ForeignNodeCreationError:
                self.logger.error(
                    f"Failed creating strength data for node {drug_product_id}"
                )
                if crash_on_error:
                    raise
                else:
                    # Continue to the next row
                    continue

            yield dc.ForeignNodePrototype(
                identifier=drug_product_id,
                strength_data=node_strength,
                dose_form=A.dose_form.get(dc.ConceptCodeVocab(row[2], vocab)),
                brand_name=A.brand_name.get(dc.ConceptCodeVocab(row[3], vocab)),
                supplier=A.supplier.get(dc.ConceptCodeVocab(row[4], vocab)),
            )

    def get_concept_strength(
        self, drug_id: dc.ConceptCodeVocab
    ) -> Sequence[dc.BoundForeignStrength]:
        """
        Extract strength combinations for a given drug concept.
        """
        strength_data = (
            self.dss.collect()
            .filter(pl.col("drug_concept_code") == drug_id.concept_code)
            .select(pl.all().exclude("drug_concept_code"))
        )

        bfs: dc.BoundForeignStrength
        if len(strength_data) == 0:
            # Return ingredient only
            ing_ir = (
                self.ir.collect()
                .filter(pl.col("concept_code_1") == drug_id.concept_code)
                .join(
                    other=self.dcs.collect().filter(
                        pl.col("concept_class_id") == "Ingredient"
                    ),
                    left_on="concept_code_2",
                    right_on="concept_code",
                    how="semi",
                )["concept_code_2"]
            )

            if len(ing_ir) == 0:
                raise ForeignNodeCreationError(
                    f"No strength nor ingredient data found for drug {drug_id}."
                )

            ingredient_data: Sequence[
                tuple[dc.Ingredient[dc.ConceptCodeVocab], None]
            ] = []
            for ingredient_concept_code in ing_ir:
                try:
                    ingredient = self.source_atoms.ingredient[
                        dc.ConceptCodeVocab(
                            ingredient_concept_code, drug_id.vocabulary_id
                        )
                    ]
                except KeyError:
                    raise ForeignNodeCreationError(
                        f"Ingredient with code {ingredient_concept_code} not "
                        f"found for drug {drug_id}."
                    )
                else:
                    bfs = (ingredient, None)
                    ingredient_data.append(bfs)

            return ingredient_data

        ingredient_concept_code: str
        strength_combinations: list[dc.BoundForeignStrength] = []
        for ingredient_concept_code, *strength in strength_data.iter_rows():
            id = dc.ConceptCodeVocab(
                ingredient_concept_code, drug_id.vocabulary_id
            )
            try:
                ingredient = self.source_atoms.ingredient[id]
            except KeyError:
                raise ForeignNodeCreationError(
                    f"Ingredient with code {ingredient_concept_code} not found "
                    f"for drug {drug_id}."
                )
            strength_combinations.append((
                ingredient,
                dc.ForeignStrength._make(strength),
            ))

        return strength_combinations
