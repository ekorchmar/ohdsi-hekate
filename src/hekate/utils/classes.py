"""Helper classes"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison
from collections.abc import Iterable
from typing import override


class SortedTuple[T: SupportsRichComparison](tuple[T]):
    """Tuple that sorts input arguments before creating self"""

    def __new__(cls, iterable: Iterable[T]):
        # Sort the iterable and pass it to the tuple constructor
        sorted_iterable = sorted(iterable)
        return super().__new__(cls, sorted_iterable)

    @override
    def __repr__(self):
        return f"SortedTuple({super().__repr__()})"
