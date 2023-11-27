import numpy as np
import pandas as pd


def novametric(df: pd.DataFrame) -> float:
    """Compute performance metric.

    :param df: Dataframe with target ('phase') and 'prediction', and identifiers.
    :return: Performance metric
    """
    df = df.copy()
    assert (
        "monthly" in df.columns
    ), "Missing 'monthly' column, only available in the train set"
    assert (
        "phase" in df.columns
    ), "Missing 'phase' column, only available in the train set"
    assert "prediction" in df.columns, "Missing 'prediction' column with predictions"

    df["date"] = pd.to_datetime(df["date"])

    # create datetime columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Sum of phasing country-brand-month = 1
    df["sum_pred"] = df.groupby(["year", "month", "brand", "country"], observed=True)[
        "prediction"
    ].transform("sum")
    assert np.isclose(
        df["sum_pred"], 1.0, rtol=1e-04
    ).all(), "Condition phasing year-month-brand-country must sum 1 is not fulfilled"

    # define quarter weights
    df["quarter_w"] = np.where(
        df["quarter"] == 1,
        1,
        np.where(df["quarter"] == 2, 0.75, np.where(df["quarter"] == 3, 0.66, 0.5)),
    )

    # compute and return metric
    return round(
        np.sqrt(
            (1 / len(df))
            * sum(
                ((df["phase"] - df["prediction"]) ** 2)
                * df["quarter_w"]
                * df["monthly"]
            )
        ),
        8,
    )
