"""
Contains implementations to read TSV data from a file and transform it
into ForeignDrugNode objects for evaluation.
"""

from abc import ABC
from csv_read.generic import CSVReader
import polars as pl  # For type hinting and schema definition


class SourceTable[IdS: pl.DataFrame | None](CSVReader[IdS], ABC):
    """
    Abstract class for reading BuildRxE input tables in CSV/TSV format.


    Attributes:
        TABLE_SCHEMA: Schema for the table.
        TABLE_COLUMNS: Ordered sequence of columns to keep from the table.
    """
