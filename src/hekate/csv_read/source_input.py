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
from csv_read.athena import OMOPVocabulariesV5
from rx_model import drug_classes as dc
from rx_model import hierarchy as h
from utils.classes import SortedTuple
from utils.constants import PERCENT_CONCEPT_ID, STRENGTH_CONFIGURATIONS_CODE
from utils.exceptions import (
    ForeignNodeCreationError,
    UnmappedSourceConceptError,
)
from utils.logger import LOGGER

type _ComponentPermutations = list[
    dc.BoundStrength[dc.ConceptCodeVocab, dc.Strength]
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
        "box_size",  # Unused for now
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


class RelationshipToConcept(SourceTable[pl.DataFrame]):
    TABLE_SCHEMA: Schema = {
        "concept_code_1": pl.Utf8,
        "vocabulary_id_1": pl.Utf8,
        "concept_id_2": pl.UInt32,
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


class PCSStage(SourceTable[pl.DataFrame]):
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
        athena_vocab: OMOPVocabulariesV5,
        delimiter: str = "\t",
        quote_char: str = '"',
    ) -> None:
        self.data_path: Path = data_path

        self.logger: logging.Logger = LOGGER.getChild(self.__class__.__name__)

        # Remember the target hierarchy
        self.target_hierarchy: OMOPVocabulariesV5 = athena_vocab

        # Initiate containers
        self.source_atoms: h.Atoms[dc.ConceptCodeVocab] = h.Atoms(self.logger)
        self.rx_atoms: h.Atoms[dc.ConceptId] = athena_vocab.atoms
        self.translator: h.NodeTranslator = h.NodeTranslator(
            rx_atoms=self.rx_atoms, logger=self.logger
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
            data_path / "relationship_to_concept.tsv",
            reference_data=self.dcs.collect().select(
                "concept_code", "vocabulary_id"
            ),
        )

        self.translator.populate_from_frame(
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

        # WARN: temporarily cleaning up all concepts with 0 in amount_value
        # There is actually a valid use case for this for drug packs, but we are
        # not processing them yet.
        zero_amounts = self.dss.collect().filter(pl.col("amount_value") == 0)
        if len(zero_amounts) > 0:
            self.logger.warning(
                f"Found {len(zero_amounts)} drugs with 0 amount_value. "
                "These will be ignored for now."
            )
        self.dss.anti_join(
            zero_amounts,
            left_on="drug_concept_code",
            right_on="drug_concept_code",
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
        pcs = PCSStage(
            data_path / "pc_stage.tsv",
            reference_data=self.dcs.collect().select("concept_code"),
        )
        if len(pcs.collect()) > 0:
            self.logger.warning(
                f"Found {len(pcs.collect())} pack_concepts. "
                "These will be ignored for now."
            )
        self.dcs.anti_join(
            pcs.collect(),
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

    def build_drug_nodes(
        self, crash_on_error: bool = False
    ) -> Generator[
        dc.ForeignDrugNode[dc.ConceptCodeVocab, dc.Strength | None], None, None
    ]:
        """
        Build DrugNodes using the DSStage and InternalRelationshipStage data.
        """
        ir = self.ir.collect()
        dcs = self.dcs.collect()

        # First, get the unique attribute data
        drug_products = dcs.filter(
            pl.col("concept_class_id") == "Drug Product"
        ).select("concept_code", "vocabulary_id")
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
            df = A.dose_form.get(dc.ConceptCodeVocab(row[2], vocab))
            bn = A.brand_name.get(dc.ConceptCodeVocab(row[3], vocab))
            sp = A.supplier.get(dc.ConceptCodeVocab(row[4], vocab))

            strength_data = self.get_strength_combinations(drug_product_id)

            # Getting strength combinations is fallible, so we need to
            # catch exceptions and log them instead of using for-loop
            total_combinations = 0
            while True:
                try:
                    strength_combination = next(strength_data)

                except StopIteration:
                    # Generator exhausted
                    break

                except ForeignNodeCreationError as e:
                    # Combination is invalid
                    if crash_on_error:
                        raise e
                    self.logger.error(
                        f"Error while generating node for {drug_product_id}: "
                        f"{e}"
                    )
                    continue

                total_combinations += 1
                yield dc.ForeignDrugNode(
                    identifier=drug_product_id,
                    strength_data=strength_combination,
                    dose_form=df,
                    brand_name=bn,
                    supplier=sp,
                )
                self.logger.debug(
                    f"Generated node #{total_combinations} for "
                    f"{drug_product_id} with {strength_combination}"
                )
            self.logger.debug(
                f"Generated {total_combinations} nodes for {drug_product_id}"
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
        self.logger.debug(
            f"Getting strength combinations for drug product {drug_product_id}"
        )
        strength_data = (
            self.dss.collect()
            .filter(pl.col("drug_concept_code") == drug_product_id.concept_code)
            .select(pl.all().exclude("drug_concept_code"))
        )

        # For drugs without strength data, just return the ingredients from
        # the IRS
        if len(strength_data) == 0:
            yield self._get_strength_ingredients_only(drug_product_id)
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
            comp_ingredient = self.source_atoms.ingredient[
                dc.ConceptCodeVocab(
                    row[0],
                    drug_product_id.vocabulary_id,
                )
            ]

            match configuration:
                case "amount_only":
                    amount: float = row[1]
                    unit: h.PseudoUnit = row[2]
                    am_permut = self.translator.translate_strength_measure(
                        amount, unit
                    )
                    component_permutations.append([
                        (comp_ingredient, dc.SolidStrength(scaled_v, true_unit))
                        for scaled_v, true_unit in am_permut
                    ])

                case "liquid_concentration":
                    numerator: float = row[3]
                    numerator_unit: h.PseudoUnit = row[4]
                    # Denominator is implicit 1
                    denominator_unit: h.PseudoUnit = row[6]
                    num_permut = self.translator.translate_strength_measure(
                        numerator, numerator_unit
                    )
                    den_permut = self.translator.translate_strength_measure(
                        1, denominator_unit
                    )
                    component_permutations.append([
                        (
                            comp_ingredient,
                            dc.LiquidConcentration(
                                numerator_value=scaled_n / scaled_d,
                                numerator_unit=n_unit,
                                denominator_unit=d_unit,
                            ),
                        )
                        for (scaled_n, n_unit), (scaled_d, d_unit) in product(
                            num_permut, den_permut
                        )
                    ])

                case "liquid_quantity":
                    numerator: float = row[3]
                    numerator_unit: h.PseudoUnit = row[4]
                    denominator: float = row[5]
                    denominator_unit: h.PseudoUnit = row[6]
                    num_permut = self.translator.translate_strength_measure(
                        numerator, numerator_unit
                    )
                    den_permut = self.translator.translate_strength_measure(
                        denominator, denominator_unit
                    )
                    component_permutations.append([
                        (
                            comp_ingredient,
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
                    ])
                case "gas_concentration":
                    # Those are static
                    numerator: float = row[3]
                    pct = self.rx_atoms.unit[dc.ConceptId(PERCENT_CONCEPT_ID)]
                    only_component = (
                        comp_ingredient,
                        dc.GasPercentage(numerator, pct),
                    )
                    component_permutations.append([only_component])

                case "wrong":
                    raise ForeignNodeCreationError(
                        "Strength data does not match any configuration."
                    )

            # Yield all possible permutations of strength components
        for combination in product(*component_permutations):
            yield SortedTuple(combination)

    def _get_strength_ingredients_only(
        self, drug_product_id: dc.ConceptCodeVocab
    ) -> SortedTuple[dc.BoundStrength[dc.ConceptCodeVocab, dc.Strength | None]]:
        """
        Get a SortedTuple of ingredients only for a drug product without
        any strength data, with None as stand-in.
        """
        ingredient_codes = (
            self.ir.collect()
            .filter(pl.col("concept_code_1") == drug_product_id.concept_code)
            .join(
                self.dcs.collect().filter(
                    pl.col("concept_class_id") == "Ingredient"
                ),
                left_on="concept_code_2",
                right_on="concept_code",
            )
            .select(
                concept_code="concept_code_2",
                vocabulary_id="vocabulary_id",
            )
        )

        ingredient_only: list[
            tuple[dc.Ingredient[dc.ConceptCodeVocab], None]
        ] = []
        for row in ingredient_codes.iter_rows():
            concept_code: str = row[0]
            vocab_id: str = row[1]
            ingredient = self.source_atoms.ingredient[
                dc.ConceptCodeVocab(concept_code, vocab_id)
            ]
            ingredient_only.append((ingredient, None))

        if len(ingredient_only) == 0:
            raise ForeignNodeCreationError(
                f"Drug {drug_product_id} has no strength data nor ingredients."
            )

        return SortedTuple(ingredient_only)

    def map_to_rxn(self) -> dict[dc.ConceptCodeVocab, list[dc.ConceptId]]:
        """
        Map the generated nodes to RxNorm concepts.
        """

        result: dict[dc.ConceptCodeVocab, list[dc.ConceptId]] = {}

        # 2 billion is conventionally used for loval concept IDs
        def new_concept_id():
            two_bill = 2_000_000_000
            while True:
                yield dc.ConceptId(two_bill)
                two_bill += 1

        cid_counter = new_concept_id()
        for node in self.build_drug_nodes(crash_on_error=False):
            translated_nodes = self.translator.translate_node(
                node, lambda: next(cid_counter)
            )
            while True:
                try:
                    option = next(translated_nodes)
                except StopIteration:
                    break
                except UnmappedSourceConceptError as e:
                    # This is expected for now
                    # TODO: skip all permutations of the node
                    self.logger.error(
                        f"Node {node.identifier} could not be mapped to "
                        f"RxNorm: {e}"
                    )
                    continue

                visitor = h.DrugNodeFinder(
                    option, self.target_hierarchy.hierarchy, self.logger
                )
                visitor.start_search()
                try:
                    node_result = visitor.get_search_results()
                except NotImplementedError:
                    # This is expected for now
                    self.logger.warning(
                        f"At least one valid Node for {option.identifier} could "
                        f"not be disambiguated."
                    )
                else:
                    result.setdefault(node.identifier, []).extend(
                        node.identifier for node in node_result.values()
                    )
        return result
