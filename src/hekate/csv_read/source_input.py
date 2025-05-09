"""
Contains implementations to read TSV data from a file and transform it
into ForeignDrugNode objects for evaluation.
"""

from __future__ import annotations

import logging
from collections.abc import (
    Generator,
    Sequence,
)  # for type annotations
from pathlib import Path  # for CSV locations
from typing import NamedTuple, override

import polars as pl  # for tabular data manipulation
from csv_read.generic import CSVReader, Schema  # Inheriting
from rx_model import drug_classes as dc  # for drug atoms and identifiers
from rx_model import hierarchy as h  # For hierarchy and atom containers
from utils.exceptions import (
    ForeignDosageStrengthError,
    ForeignNodeCreationError,
)
from utils.classes import (
    # For type aliases
    PlConceptId,
    PlRealNumber,
    PyRealNumber,
    SortedTuple,
    PlString,
    PlSmallInt,
)
from utils.logger import LOGGER

type _SourceBoxSize = int | None


class _DrugViewRow(NamedTuple):
    concept_code: str
    vocabulary_id: str
    dose_form_code: str | None
    brand_name_code: str | None
    supplier_code: str | None


class _PCSViewRow(NamedTuple):
    concept_code: str
    vocabulary_id: str
    drug_concept_code: list[str]
    amount: list[int | None]
    box_size: list[int] | list[None]
    brand_name_code: str | None
    supplier_code: str | None


class DrugConceptStage(CSVReader[None]):
    TABLE_SCHEMA: Schema = {
        "concept_code": PlString,
        "concept_name": PlString,
        "concept_class_id": PlString,
        "vocabulary_id": PlString,
        "source_concept_class_id": PlString,
        "possible_excipient": pl.Null,  # NOTE: not implemented yet
        "valid_start_date": pl.Date,
        "valid_end_date": pl.Date,
        "invalid_reason": PlString,
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
    type dss_strength_tuple = tuple[PyRealNumber | None, dc.PseudoUnit]

    TABLE_SCHEMA: Schema = {
        "drug_concept_code": PlString,
        "ingredient_concept_code": PlString,
        "amount_value": PlRealNumber,
        "amount_unit": PlString,
        "numerator_value": PlRealNumber,
        "numerator_unit": PlString,
        "denominator_value": PlRealNumber,
        "denominator_unit": PlString,
        "box_size": PlSmallInt,
    }

    TABLE_COLUMNS: list[str] = [
        "drug_concept_code",
        "ingredient_concept_code",
        # Use matching order for strength tuple
        *dc.ForeignStrength._fields,
        "box_size",
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
        "concept_code_1": PlString,
        "vocabulary_id_1": PlString,
        "concept_id_2": PlConceptId,
        "precedence": PlSmallInt,
        "conversion_factor": PlRealNumber,
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
        "concept_code_1": PlString,
        "concept_code_2": PlString,
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
        "pack_concept_code": PlString,
        "drug_concept_code": PlString,
        "amount": PlSmallInt,
        "box_size": PlSmallInt,
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

        self.pcs: PCSStage = PCSStage(
            data_path / "pc_stage.tsv",
            reference_data=self.dcs.collect().select("concept_code"),
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
    ) -> Generator[dc.ForeignNodePrototype]:
        """
        Build Drug node prototypes using the DSStage and
        InternalRelationshipStage data.
        """
        ir = self.ir.collect()
        dcs = self.dcs.collect()

        # First, get the unique attribute data
        drug_products = (
            dcs.filter(pl.col("concept_class_id") == "Drug Product")
            .join(
                # Exclude pack nodes
                other=self.pcs.collect(),
                left_on="concept_code",
                right_on="pack_concept_code",
                how="anti",
            )
            .join(
                # Exclude Drug Products having explicit mappings in RTC
                # TODO: Should rise a warning
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
                drug_products = drug_products.join(
                    other=ir_of_attr,
                    left_on="concept_code",
                    right_on="concept_code_1",
                    how="left",
                    validate="1:1",  # TODO: Make this an external QA check
                ).rename({"concept_code_2": field_name + "_code"})
            except pl.exceptions.ComputeError as e:
                if crash_on_error:
                    raise e
                self.logger.error(
                    f"Error while validating uniqueness of {attr_class} data "
                    f"for Drug Products: {e}"
                )

        for row in drug_products.iter_rows():
            drug_data = _DrugViewRow._make(row)
            drug_product_id = dc.ConceptCodeVocab(
                drug_data.concept_code,
                drug_data.vocabulary_id,
            )

            try:
                strength, box_size = self.get_concept_strength(drug_product_id)
            except ForeignNodeCreationError as e:
                self.logger.error(
                    f"Failed creating strength data for node {drug_product_id}:"
                    f" {e}"
                )
                if crash_on_error:
                    raise
                else:
                    # Continue to the next row
                    continue

            yield dc.ForeignNodePrototype(
                identifier=drug_product_id,
                strength_data=SortedTuple(strength),
                dose_form=self.source_atoms.dose_form.get(
                    dc.ConceptCodeVocab(
                        drug_data.dose_form_code,
                        drug_data.vocabulary_id,
                    )
                )
                if drug_data.dose_form_code
                else None,
                brand_name=self.source_atoms.brand_name.get(
                    dc.ConceptCodeVocab(
                        drug_data.brand_name_code,
                        drug_data.vocabulary_id,
                    )
                )
                if drug_data.brand_name_code
                else None,
                supplier=self.source_atoms.supplier.get(
                    dc.ConceptCodeVocab(
                        drug_data.supplier_code,
                        drug_data.vocabulary_id,
                    )
                )
                if drug_data.supplier_code
                else None,
                box_size=box_size,
            )

    def prepare_pack_nodes(
        self,
        crash_on_error: bool = False,
    ) -> Generator[dc.ForeignPackNodePrototype]:
        """
        Generate ForeignPackNodePrototypes from PACK_CONTENT data and provided
        drug content node translations
        """
        ir = self.ir.collect()
        dcs = self.dcs.collect()

        # First, get the unique attribute data
        packs = (
            dcs.filter(pl.col("concept_class_id") == "Drug Product")
            .join(
                # Pack nodes
                other=self.pcs.collect(),
                left_on="concept_code",
                right_on="pack_concept_code",
            )
            .join(
                # Exclude Drug Products having explicit mappings in RTC
                # TODO: Should rise a warning
                other=self.rtcs.collect(),
                left_on=["concept_code", "vocabulary_id"],
                right_on=["concept_code_1", "vocabulary_id_1"],
                how="anti",
            )
            .select(
                "concept_code",
                "vocabulary_id",
                "drug_concept_code",
                "amount",
                "box_size",
            )
            # Nest pack entry data into lists
            .group_by("concept_code", "vocabulary_id")
            .all()
        )

        for attr_class in ["Brand Name", "Supplier"]:
            ir_of_attr = ir.join(
                other=dcs.filter(pl.col("concept_class_id") == attr_class),
                left_on="concept_code_2",
                right_on="concept_code",
                how="semi",
            )

            field_name = attr_class.lower().replace(" ", "_")
            try:
                packs = packs.join(
                    other=ir_of_attr,
                    left_on="concept_code",
                    right_on="concept_code_1",
                    how="left",
                    validate="1:1",  # TODO: Make this an external QA check
                ).rename({"concept_code_2": field_name + "_code"})

            except pl.exceptions.ComputeError as e:
                if crash_on_error:
                    raise e
                self.logger.error(
                    f"Error while validating uniqueness of {attr_class} data "
                    f"for Drug Products: {e}"
                )

        for row in packs.iter_rows():
            pack_data = _PCSViewRow._make(row)

            pack_id = dc.ConceptCodeVocab(
                pack_data.concept_code,
                pack_data.vocabulary_id,
            )

            source_entries: zip[
                tuple[dc.ConceptCodeVocab, int | None, int | None]
            ] = zip(
                (
                    dc.ConceptCodeVocab(drug_code, pack_data.vocabulary_id)
                    for drug_code in pack_data.drug_concept_code
                ),
                pack_data.amount,
                pack_data.box_size,
            )

            brand_name = (
                self.source_atoms.brand_name.get(
                    dc.ConceptCodeVocab(
                        pack_data.brand_name_code,
                        pack_data.vocabulary_id,
                    )
                )
                if pack_data.brand_name_code
                else None
            )

            supplier = (
                self.source_atoms.supplier.get(
                    dc.ConceptCodeVocab(
                        pack_data.supplier_code,
                        pack_data.vocabulary_id,
                    )
                )
                if pack_data.supplier_code
                else None
            )

            entries_prototype = [
                dc.ForeignPackEntryPrototype._make(entry_tuple)
                for entry_tuple in source_entries
            ]

            yield dc.ForeignPackNodePrototype(
                identifier=pack_id,
                entries=tuple(entries_prototype),
                brand_name=brand_name,
                supplier=supplier,
            )

    def get_concept_strength(
        self, drug_id: dc.ConceptCodeVocab
    ) -> tuple[Sequence[dc.BoundForeignStrength], _SourceBoxSize]:
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
            ing_ir: list[str] = (
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
            ).to_list()

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

            return ingredient_data, None

        ingredient_concept_code: str
        strength_combinations: list[dc.BoundForeignStrength] = []
        box_size: _SourceBoxSize
        box_sizes: set[_SourceBoxSize] = set()
        for (
            ingredient_concept_code,
            *strength,
            box_size,
        ) in strength_data.iter_rows():
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
            box_sizes.add(box_size)

        if len(box_sizes) > 1:
            raise ForeignDosageStrengthError(
                f"Multiple box sizes found for drug {drug_id}: {box_sizes}."
            )

        return strength_combinations, box_sizes.pop()
