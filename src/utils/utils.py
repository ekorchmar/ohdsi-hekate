"""Snippets for utils."""
from typing import TypeVar

T, U = TypeVar("T"), TypeVar("U")


def keep_multiple_values(data: dict[T, U]) -> dict[T, U]:
    """Filter the dictionary to keep only the values that are duplicated."""
    return {k: v for k, v in data.items() if list(data.values()).count(v) > 1}


def invert_dict(data: dict[T, U]) -> dict[U, list[T]]:
    """Invert the dictionary, where unique values become keys, and list of
    former keys become values."""
    return {v: k for k, v in data.items()}
