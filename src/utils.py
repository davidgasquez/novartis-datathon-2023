"""
Functions to make our life easier.
"""

from datetime import datetime

import numpy as np
import pandas as pd


def postprocess(
    df,
    zero_threshold=None,
    min_max_scale=False,
    absolute_minimum=False,
    train_data=None,
) -> pd.DataFrame:
    local_df = df.copy()

    local_df["prediction"] = np.expm1(local_df["prediction"])

    if train_data is not None:
        local_train_data = train_data[["brand", "country", "month", "phase"]].copy()
        cbm_q = (
            train_data.groupby(["brand", "country", "month"], observed=True)["phase"]
            .apply(lambda x: (x == 0).sum() / len(x))
            .reset_index()
            .rename(columns={"phase": "cbm_q"})
        )

        local_train_data = local_train_data.merge(
            cbm_q, on=["brand", "country", "month"], how="left"
        )

        # Compute the phase
        cbm_0_phase = (
            local_train_data.groupby(["brand", "country", "month"], observed=True)
            .apply(lambda c: c["phase"].quantile(c["cbm_q"].mean()))
            .reset_index()
            .rename(columns={0: "cbm_0_phase"})
        )

        local_df = local_df.merge(
            cbm_0_phase, on=["brand", "country", "month"], how="left"
        )

        local_df["prediction"] = local_df["prediction"] - local_df[
            "cbm_0_phase"
        ].fillna(0)

    if min_max_scale:
        # Substract the min to make it 0
        local_df["prediction"] = local_df["prediction"] - local_df["prediction"].min()
        # Scale by max to make it 1
        local_df["prediction"] = local_df["prediction"] / local_df["prediction"].max()

    if absolute_minimum:
        # Set negative predictions to 0
        local_df["prediction"] = local_df["prediction"] + abs(
            local_df["prediction"].min()
        )

    # Set negative predictions to 0
    if zero_threshold is not None:
        local_df["prediction"] = local_df["prediction"].apply(
            lambda x: 0 if x < zero_threshold else x
        )

    # Normalize predictions to sum to 1
    local_df["prediction"] = (
        local_df.groupby(["brand", "country", "year", "month"], observed=True)[
            "prediction"
        ]
        .transform(lambda x: x / x.sum())
        .fillna(0)
    )

    return local_df


def save_submission(
    dataframe, name="submission", p=0.0, path: str = "../data/submissions/"
):
    """Save submission to csv file with the current timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    file_name = str(name) + "_" + timestamp + "_" + f"{p}" + ".csv"
    dataframe.to_csv(path + file_name, index=False)

    return file_name


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":  # for integers
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:  # for floats.
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
