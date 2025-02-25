"""Helper classes"""
from dataclasses import dataclass


class SortedTuple(tuple):
    """Tuple that sorts input arguments before creating self"""
    def __new__(cls, iterable):
        # Sort the iterable and pass it to the tuple constructor
        sorted_iterable = sorted(iterable)
        return super().__new__(cls, sorted_iterable)

    def __repr__(self):
        return f"SortedTuple({super().__repr__()})"


elementary_dataclass = dataclass(frozen=True, order=True, eq=True, slots=True)
complex_dataclass = dataclass(frozen=True, eq=True, slots=True)
