"""
Generic CSV reader, able to iterate over rows of Athena or
BuildRxE input CSV files.
"""

import logging
from abc import ABC
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable
from os.path import expandvars

import polars as pl
from utils.logger import LOGGER

type Schema = Mapping[str, type[pl.DataType] | pl.Decimal]
type LineFilter[D: pl.DataFrame | pl.Series | pl.LazyFrame | None] = Callable[
    [pl.LazyFrame, D], pl.LazyFrame
]


class CSVReader[D: pl.DataFrame | pl.Series | pl.LazyFrame | None](ABC):
    """
    Generic CSV reader, able to read CSV or TSV files in batches and iterate
    over their contents.
    """

    TABLE_SCHEMA: Schema
    TABLE_COLUMNS: list[str]

    def table_filter(
        self, frame: pl.LazyFrame, valid_concepts: D = None
    ) -> pl.LazyFrame:
        """
        Filter function to apply to the table.

        Modifications to polars.LazyFrame to optionally
        discard or otherwise modify rows. Defaults to noop. The
        function should take a `pl.LazyFrame` as input and return a
        `pl.LazyFrame`. Line filters may require an additional argument,
        to provide known valid identifiers, which can be passed using the
        `valid_concepts` parameter.

        Read more at https://docs.pola.rs/user-guide/concepts/lazy-api/
        """
        del valid_concepts
        return frame

    def __init__(
        self,
        path: Path,
        delimiter: str = "\t",
        quote_char: str | None = None,
        reference_data: D = None,
    ):
        """
        Args:
            path: Path to the CSV file.

            delimiter: Delimiter used in the CSV file. Defaults to "\\t" as is
                historically used by Athena.

            quote_char: Optional character used to quote fields. Defaults to
                `None` to disable all quote processnig, as is historically
                unused by Athena.

            reference_data: Optional argument to pass to the `line_filter`
                function. resulting call will look like `line_filter(lazy_frame,
                reference_data)` This is almost always going to be a DataFrame
                or Series of key field identifiers.
        """
        self.delimiter: str = delimiter
        self.path: Path = path

        # Associate a logger with the pathname
        self.logger: logging.Logger = LOGGER.getChild(
            self.__class__.__name__
        ).getChild(path.name)

        # Create a reader object
        self._lazy_frame: pl.LazyFrame = pl.scan_csv(
            source=expandvars(self.path),
            separator=delimiter,
            quote_char=quote_char,
            has_header=True,
            # NOTE: supplying a `schema` parameter causes Polars to expect a
            # certain column ordering, which is not guaranteed from BuildRxE
            # input files. We use implicitly unordered `schema_overrides`
            # instead.
            schema_overrides=self.TABLE_SCHEMA,
            infer_schema_length=0,  # We don't really need to infer anything
        )

        # Apply the filtering
        self._lazy_frame = self.table_filter(
            self._lazy_frame, reference_data
        ).select(self.TABLE_COLUMNS)

        self.logger.debug(self._lazy_frame.explain())

        self.data: pl.DataFrame | None = None

        self.logger.info(f"Preparing to read from {path.name}")

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
