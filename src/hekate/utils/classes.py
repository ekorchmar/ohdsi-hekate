"""Helper classes"""

import decimal
import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison
from collections.abc import Iterable
from typing import override

import polars as pl


class SortedTuple[T: SupportsRichComparison](tuple[T]):
    """Tuple that sorts input arguments before creating self"""

    def __new__(cls, iterable: Iterable[T]):
        # Sort the iterable and pass it to the tuple constructor
        sorted_iterable = sorted(iterable)
        return super().__new__(cls, sorted_iterable)

    @override
    def __repr__(self):
        return f"SortedTuple({super().__repr__()})"


class Cardinality(enum.Enum):
    """
    Enum to define the cardinality of a relationship between two concepts

    Left hand side (source concept, concept_id_1) is always assumed to have
    cardinality of 1. Cardinality counts are always in relation to the target,
    showing how many target concepts can be related to a single source concept.
    """

    ANY = "0..*"  # Will not be used in practice
    ONE = "1..1"
    OPTIONAL = "0..1"
    NONZERO = "1..*"


PlRealNumber = pl.Decimal(
    precision=None,  # infer
    scale=6,  # Down to 6 decimal places
)
PyRealNumber = decimal.Decimal

CARDINALITY_REQUIRED = [Cardinality.ONE, Cardinality.NONZERO]
CARDINALITY_SINGLE = [Cardinality.ONE, Cardinality.OPTIONAL]
