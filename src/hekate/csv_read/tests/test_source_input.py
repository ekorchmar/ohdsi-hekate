import pytest
import polars as pl
import datetime
from csv_read.source_input import DrugConceptStage


def test_date_to_yyyymmdd():
    if pl.__version__.split(".") <= ["1", "23", "0"]:
        pytest.skip("Broken upstream")

    df = pl.DataFrame(
        data={
            "dt": (
                "1970-01-01",
                "2099-12-31",
                "2021-05-31",
                "1994-09-29",
                "1995-10-12",
            )
        },
        schema={"dt": pl.Date},
    )

    result = df.select(dt=DrugConceptStage.date_to_yyyymmdd("dt"))["dt"]

    assert all(
        result
        == pl.Series(
            [
                1970_01_01,
                2099_12_31,
                2021_05_31,
                1994_09_29,
                1995_10_12,
            ],
            dtype=pl.Int32,
        )
    )
