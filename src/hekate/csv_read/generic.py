"""
Generic CSV reader, able to iterate over rows of Athena or
BuildRxE input CSV files.
"""

from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Callable, TypeVar
from collections.abc import Sequence

import polars as pl

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
        keep_columns: Sequence[str],
        delimiter: str = "\t",
        quote_char: str | None = None,
        line_filter: LineFilter[T] | None = None,
        filter_arg: T | None = None,
    ):
        """
        Args:
            path: Path to the CSV file.

            schema: Obligatory schema to use when reading the CSV file.

            keep_columns: List of column names to keep. Defaults to `None` to
                include all columns. If filter is specified, columns will be
                applied after filtering.

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
                resulting call will look like `line_filter(lazy_frame,
                filter_arg)`.

            schema: Required schema to use when reading the CSV file. Provided
                as a dictionary of column names and their respective Polars
                data types.

        """
        self.delimiter: str = delimiter
        self.path: Path = path
        self.schema: Schema = schema
        self.columns: Sequence[str] = keep_columns

        # Associate a logger with the pathname
        self.logger: logging.Logger = LOGGER.getChild(
            self.__class__.__name__
        ).getChild(path.name)

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

        # Apply column selection
        self._lazy_frame = self._lazy_frame.select(self.columns)

        self.logger.debug(self._lazy_frame.explain())

        self.data: pl.DataFrame | None = None

    def materialize(self):
        """
        Materialize the LazyFrame into a DataFrame. All filters from the
        lazy frame are applied on read, skipping rows and columns and applying
        dynamic calculations.
        """
        if self.data is not None:
            return
        self.logger.info("Collecting and caching data from disk")
        self.data = self._lazy_frame.collect()
        assert self.data is not None
        self.logger.info(f"{len(self.data):,} rows collected and cached")

    def collect(self) -> pl.DataFrame:
        """
        Collect the entire CSV file into a DataFrame.
        """
        if self.data is None:
            self.materialize()
            assert self.data is not None
        return self.data

    def filter(self, *predicates: pl.Expr) -> None:
        """
        Filter the data materialized data using a predicate. This is a
        destructive operation that changes the materialized data in place.

        Filter may be applied before data is materialized, in which case
        it will be applied to the lazy frame.
        """
        if self.data is None:
            self.logger.info("Filtering data before materialization")
            self._lazy_frame = self._lazy_frame.filter(*predicates)

        else:
            old_len = len(self.data)
            self.data = self.data.filter(*predicates)
            self.logger.info(
                f"Filtered {old_len - len(self.data):,} rows from "
                f"{old_len:,} with predicate"
            )

    def anti_join(
        self,
        other: pl.DataFrame,
        *,
        on: Sequence[str] | None = None,
        left_on: Sequence[str] | None = None,
        right_on: Sequence[str] | None = None,
    ) -> None:
        """
        Anti-join the data with another DataFrame. This is a destructive
        operation that changes the materialized data in place.

        Anti-join may be applied before data is materialized, in which case
        data will be materialized first.

        Args:
            other: DataFrame to anti-join with.

            on: Columns to join on. Must be present in both DataFrames. If this
                parameter is provided, `left_on` and `right_on` must be `None`.

            left_on: Columns to join on in the left DataFrame. Must be present
                in the left DataFrame. If this parameter is provided, `on` must
                be `None` and `right_on` must be provided.

            right_on: Columns to join on in the right DataFrame. Must be present
                in the right DataFrame. If this parameter is provided, `on` must
                be `None` and `left_on` must be provided.
        """
        if self.data is None:
            self.materialize()
            assert self.data is not None

        old_len = len(self.data)

        self.data = self.data.join(
            other=other,
            how="anti",
            on=on,
            left_on=left_on,
            right_on=right_on,
        )

        self.logger.info(
            f"Removed {old_len - len(self.data):,} rows with anti-join of "
            f"{len(other):,} rows"
        )
