"""Snippets for utils."""

from collections import Counter
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


def keep_multiple_values(data: dict[T, U]) -> dict[T, U]:
    """Filter the dictionary to keep only the values that are duplicated."""
    return {k: v for k, v in data.items() if list(data.values()).count(v) > 1}


def count_repeated_first_entries(data: Iterable[tuple[T, U]]) -> dict[T, int]:
    """Count the number of times the first entry is repeated in the tuples."""
    counts = Counter(x[0] for x in data)
    return {k: v for k, v in counts.items() if v > 1}


def invert_merge_dict(data: dict[T, U]) -> dict[U, list[T]]:
    """Invert the dictionary, where unique values become keys, and list of
    former keys become values."""
    inverted: dict[U, list[T]] = {}
    for k, v in data.items():
        inverted.setdefault(v, []).append(k)
    return inverted
