"""\
    Generic CSV reader, able to iterate over rows of Athena or
    BuildRxE input CSV files.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import csv
import polars as pl

from src.utils.exceptions import SchemaError


class CSVReader:
    """
    Generic CSV reader, able to read CSV files in batches and iterate
    over their contents.
    """

    def __init__(
        self,
        path: Path,
        schema: Mapping[str, pl.datatypes.DataType],
        delimiter: str = "\t",
        line_filter: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
        line_processor: Callable[[tuple], None] | None = None,
        batch_size: int = 100_000,
    ):
        """
        Args:
            path: Path to the CSV file.

            schema: Obligatory schema to use when reading the CSV file.

            columns: List of column names. Defaults to `None` to include
                all columns.

            delimiter: Delimiter used in the CSV file. Defaults to "\\t" as is
                historically used by Athena.

            line_filter: Modifications to polars.LazyFrame to optionally discard
                or otherwise modify rows. Defaults to `None`. The function should
                take a `pl.LazyFrame` as input and return a `pl.LazyFrame`.

                Read more at https://docs.pola.rs/user-guide/concepts/lazy-api/

            line_processor: Optional function to process each row.
                Defaults to `None`. The function should take a `pl.Series`
                as input. No values returned by the function are saved, so
                the function should have useful side effects.

            schema: Optional schema to use when reading the CSV file.

            batch_size: Number of rows to read at a time. Defaults to
                100_000.
        """
        self.batch_size = batch_size
        self.delimiter = delimiter
        self.line_processor = line_processor
        self.path = path
        self.schema = schema

        # Find actual column layout
        self.parse_header()

        # Create a reader object
        self._lazy_frame = pl.scan_csv(
            source=self.path,
            has_header=True,
            infer_schema=False,
            schema=self.schema,
        )

        # Apply line filter if provided
        if line_filter:
            self._lazy_frame = line_filter(self._lazy_frame)

        self.data: pl.DataFrame | None = None

    def infer_header_order(self) -> None:
        """
        Parse the header to validate the known schema.
        """
        with open(self.path, "r") as file:
            reader = csv.reader(file, delimiter=self.delimiter)
            self.header: list[str] = list(next(reader))

        missed_columns = set(self.schema) - set(self.header)
        if missed_columns:
            raise SchemaError(
                f"Columns {', '.join(missed_columns)} not found in the header."
            )

    def collect(self) -> pl.DataFrame:
        """
        Collect the entire CSV file into a DataFrame.
        """
        self.data = self._lazy_frame.collect()
        return self.data

    def set_line_processor(
        self, line_processor: Callable[[tuple], None]
    ) -> None:
        """
        Set the line processor to use when processing the data.
        """
        self.line_processor = line_processor

    def process_data(self) -> None:
        """
        Process the stored data with the line processor.
        """
        if not self.line_processor:
            raise ValueError(f"No line processor provided in {self}")

        if not self.data:
            raise ValueError(f"No data stored yet to process in {self}")

        self.data.map_rows(self.line_processor)


a = CSVReader
