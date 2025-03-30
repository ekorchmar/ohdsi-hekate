"""
Contains base classes for declarative OMOP representation of RxNorm and
RxNorm Extension concepts.
"""

from utils.enums import DomainId, VocabularyId, ConceptClassId  # For enums
from dataclasses import dataclass  # For dataclass definitions

import polars as pl  # For expression definitions


RX_VOCAB = VocabularyId.RXE, VocabularyId.RXN


@dataclass(frozen=True, eq=True)
class ConceptDefinition:
    constructor: type
    omop_concept_class_id: ConceptClassId
    omop_domain_id: DomainId
    omop_vocabulary_ids: tuple[VocabularyId, ...]
    standard_concept: bool

    def get_concept_expression(self) -> pl.Expr:
        return (
            (pl.col("concept_class_id") == self.omop_concept_class_id.value)
            & (pl.col("domain_id") == self.omop_domain_id.value)
            & (
                (pl.col("standard_concept") == "S")
                if self.standard_concept
                else pl.col("standard_concept").is_null()
            )
            & (
                pl.col("vocabulary_id").is_in([
                    vocabulary_id.value
                    for vocabulary_id in self.omop_vocabulary_ids
                ])
            )
        )

    def get_abbreviation(self) -> str:
        return "".join(
            word[0].upper() for word in self.omop_concept_class_id.value.split()
        )

    def get_colname(self) -> str:
        return (
            "_".join(self.omop_concept_class_id.value.lower().split())
            + "_concept_id"
        )

    @property
    def class_id(self) -> str:
        return self.omop_concept_class_id.value
