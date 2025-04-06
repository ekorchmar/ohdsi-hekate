"""Helper classes"""

import decimal
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


# Polars types
PlConceptId = pl.UInt32
PlString = pl.Utf8
PlAthenaDate = pl.UInt32  # YYYYMMDD in Athena
PlSmallInt = pl.UInt16

PlRealNumber = pl.Decimal(
    precision=None,  # infer
    scale=6,  # Down to 6 decimal places
)
PyRealNumber = decimal.Decimal
