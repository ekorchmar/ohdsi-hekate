"""\
    Generic CSV reader, able to iterate over rows of Athena or
    BuildRxE input CSV files.
"""

from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Callable, TypeVar

import csv
import polars as pl

from utils.exceptions import SchemaError
from utils.logger import LOGGER

T = TypeVar("T")

type Schema = Mapping[str, type[pl.DataType]]
type LineFilter[T] = Callable[[pl.LazyFrame, T | None], pl.LazyFrame]


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
        line_filter: LineFilter[T] | None = None,
        filter_arg: T | None = None,
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
                `pl.LazyFrame`. Line filters may require an additional argument,
                which can be passed using the `filter_arg` parameter.

                Read more at https://docs.pola.rs/user-guide/concepts/lazy-api/

            filter_arg: Optional argument to pass to the `line_filter` function.
                resulting call will look like
                `line_filter(lazy_frame, filter_arg)`.

            schema: Required schema to use when reading the CSV file. Provided
                as a dictionary of column names and their respective Polars
                data types.

        """
        self.delimiter: str = delimiter
        self.path: Path = path
        self.schema: Schema = schema

        # Associate a logger with the pathname
        self.logger: logging.Logger = LOGGER.getChild(path.name)

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
            self._lazy_frame = line_filter(self._lazy_frame, filter_arg)
        self.logger.debug(self._lazy_frame.explain())

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
        if self.data is None:
            self.logger.info("Collecting and caching data from disk")
            self.data = self._lazy_frame.collect()
            self.logger.info(f"{len(self.data)} rows collected and cached")
        return self.data

    def filter(self, predicate: pl.Expr) -> None:
        """
        Filter the data materialized data using a predicate. This is a
        destructive operation that changes the materialized data in place.
        """
        if self.data is None:
            raise ValueError(
                "Data is not yet materialized. Have you called collect()?"
            )

        self.data = self.data.filter(predicate)
