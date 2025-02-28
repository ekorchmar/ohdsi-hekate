"""\
    Generic CSV reader, able to iterate over rows of Athena or
    BuildRxE input CSV files.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import csv
import polars as pl

from ..utils.exceptions import SchemaError

type Schema = Mapping[str, type[pl.DataType]]


class CSVReader:
    """
    Generic CSV reader, able to read CSV files in batches and iterate
    over their contents.
    """

    def __init__(
        self,
        path: Path,
        schema: Schema,
        delimiter: str = "\t",
        quote_char: str | None = None,
        line_filter: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    ):
        """
        Args:
            path: Path to the CSV file.

            schema: Obligatory schema to use when reading the CSV file.

            columns: List of column names. Defaults to `None` to include
                all columns.

            delimiter: Delimiter used in the CSV file. Defaults to "\\t" as is
                historically used by Athena.

            quote_char: Optional character used to quote fields. Defaults to
                `None` to disable all quote processnig, as is historically
                unused by Athena.

            line_filter: Modifications to polars.LazyFrame to optionally
                discard or otherwise modify rows. Defaults to `None`. The
                function should take a `pl.LazyFrame` as input and return a
                `pl.LazyFrame`.

                Read more at https://docs.pola.rs/user-guide/concepts/lazy-api/

            schema: Required schema to use when reading the CSV file. Provided
                as a dictionary of column names and their respective Polars
                data types.

        """
        self.delimiter: str = delimiter
        self.path: Path = path
        self.schema: Schema = schema

        # Find actual column layout
        self.infer_header_order()

        # Create a reader object
        self._lazy_frame: pl.LazyFrame = pl.scan_csv(
            source=self.path,
            separator=delimiter,
            quote_char=quote_char,
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
