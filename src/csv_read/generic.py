"""\
    Generic CSV reader, able to iterate over rows of Athena or
    BuildRxE input CSV files.
"""

from collections.abc import Generator
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import polars


class CSVReader:
    """
    Generic CSV reader, able to read CSV files in batches and iterate
    over their contents.
    """

    def __init__(
        self,
        path: Path,
        schema: Mapping[str, polars.datatypes.DataType],
        columns: list[str] = None,
        delimiter: str = "\t",
        line_processor: Callable[[polars.Series], None] | None = None,
        batch_size: int = 100_000,
    ):
        """
        Args:
            path: Path to the CSV file.
            schema: Obligatory schema to use when reading the CSV file.
            columns: List of column names. Defaults to `None` to include
                all columns.
            delimiter: Delimiter used in the CSV file. Defaults to "\\t" as
                historically used by Athena.
            line_processor: Optional function to process each row.
                Defaults to None. The function should take a polars.Series
                as input. No values returned by the function are saved, so
                the function should have useful side effects.
            schema: Optional schema to use when reading the CSV file.
            batch_size: Number of rows to read at a time. Defaults to
                100_000.
        """
        self.batch_size = batch_size
        self.columns = columns
        self.delimiter = delimiter
        self.line_processor = line_processor
        self.path = path
        self.schema = schema

        self.parse_header()

    def parse_header() -> None:
        """
        Parse the header to validate the known schema and to reorder the
        columns in the reader.
        """
        raise NotImplementedError

    def read_chunk(self) -> Generator[polars.DataFrame, None, None]:
        """
        Read the CSV file rows and iterate over its contents.

        Yields:
            polars.DataFrame: A DataFrame containing the rows read from
                the CSV file.
        """


a = CSVReader
