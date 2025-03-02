"""Snippets for utils."""

from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


def keep_multiple_values(data: dict[T, U]) -> dict[T, U]:
    """Filter the dictionary to keep only the values that are duplicated."""
    return {k: v for k, v in data.items() if list(data.values()).count(v) > 1}


def invert_merge_dict(data: dict[T, U]) -> dict[U, list[T]]:
    """Invert the dictionary, where unique values become keys, and list of
    former keys become values."""
    inverted: dict[U, list[T]] = {}
    for k, v in data.items():
        inverted.setdefault(v, []).append(k)
    return inverted
