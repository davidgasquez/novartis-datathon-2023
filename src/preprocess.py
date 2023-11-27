import numpy as np
import pandas as pd


def static_augment(
    df,
    date_features=True,
    laggings=False,
    monthly_lag=False,
    simple_categorial_facts=False,
    fix_dayweek_count=False,
    phase_country_stats=False,
    phase_brand_stats=False,
    phase_bcm_stats=False,
    phase_brand_month_stats=False,
    phase_country_month_stats=False,
    phase_wd_month_stats=False,
    phase_dayweek_stats=False,
    phase_wd_stats=False,
    phase_0_lag_stats=False,
    # These leak
    gamma_features=False,
    gamma_log_features=False,
    norm_features=False,
    norm_log_features=False,
    sample_weight=False,
) -> pd.DataFrame:
    local_df = df.copy()

    if date_features:
        local_df = add_date_features(local_df)

    if laggings:
        local_df = add_lagging(local_df)

    if phase_bcm_stats:
        local_df = add_phase_bcm_stats(local_df)

    if phase_brand_month_stats:
        local_df = add_phase_brand_month_stats(local_df)

    if phase_wd_month_stats:
        local_df = add_phase_wd_month_stats(local_df)

    if phase_country_stats:
        local_df = add_phase_country_stats(local_df)

    if phase_brand_stats:
        local_df = add_phase_brand_stats(local_df)

    if phase_dayweek_stats:
        local_df = add_phase_dayweek_stats(local_df)

    if phase_wd_stats:
        local_df = add_phase_wd_stats(local_df)

    if phase_country_month_stats:
        local_df = add_phase_country_month_stats(local_df)

    if simple_categorial_facts:
        local_df = add_simple_categorical_facts(local_df)

    if monthly_lag:
        local_df = add_monthly_lag(local_df)

    if fix_dayweek_count:
        local_df = fix_day_week_count(local_df)

    if gamma_features:
        local_df = add_gamma_features(local_df)

    if gamma_log_features:
        local_df = add_gamma_log_features(local_df)

    if norm_features:
        local_df = add_norm_features(local_df)

    if norm_log_features:
        local_df = add_norm_log_features(local_df)

    if phase_0_lag_stats:
        local_df = add_phase_0_lag_stats(local_df)

    if sample_weight:
        local_df = add_sample_weight(local_df)

    # Cast to Categorical
    for col in local_df.select_dtypes(include="object").columns:
        local_df[col] = local_df[col].astype("category")  # type: ignore

    return local_df


def add_wd_phase_0_proportion(df) -> pd.DataFrame:
    df.groupby(["brand", "country", "year", "month"], observed=True)[
        "dayweek"
    ].value_counts().reset_index().pivot(
        index=["brand", "country", "year", "month"], columns="dayweek", values="count"
    ).fillna(0).astype(int).reset_index().rename(
        columns={i: f"dayweek_{i}" for i in range(7)}
    )

    dayweek_phase = (
        df.groupby(["wd"])["phase"].value_counts(normalize=True).reset_index()
    )

    dayweek_phase = (
        dayweek_phase[dayweek_phase["phase"] == 0]
        .sort_values("wd")
        .drop("phase", axis=1)
        .rename(columns={"proportion": "wd_phase_0_proportion"})
    )

    return df.merge(dayweek_phase, on=["wd"], how="left")


def add_date_features(df) -> pd.DataFrame:
    local_df = df.copy()

    # Basic Date Features
    local_df["date"] = pd.to_datetime(local_df["date"])
    local_df["year"] = local_df["date"].dt.year
    local_df["month"] = local_df["date"].dt.month
    local_df["quarter"] = local_df["date"].dt.quarter
    local_df["day"] = local_df["date"].dt.day
    local_df["weekday"] = local_df["date"].dt.dayofweek
    local_df["week"] = local_df["date"].dt.isocalendar().week
    local_df["is_weekend"] = local_df["weekday"].isin([5, 6]).astype(int)

    # Sin + cos for dayweek
    local_df["weekday_sin"] = np.sin(2 * np.pi * local_df["weekday"] / 7)
    local_df["weekday_cos"] = np.cos(2 * np.pi * local_df["weekday"] / 7)

    # Sin + cos for month
    local_df["month_sin"] = np.sin(2 * np.pi * local_df["month"] / 12)
    local_df["month_cos"] = np.cos(2 * np.pi * local_df["month"] / 12)

    # Sin + cos for week
    local_df["week_sin"] = np.sin(2 * np.pi * local_df["week"] / 52)
    local_df["week_cos"] = np.cos(2 * np.pi * local_df["week"] / 52)

    # Sin + cos for day
    local_df["day_sin"] = np.sin(2 * np.pi * local_df["day"] / 31)
    local_df["day_cos"] = np.cos(2 * np.pi * local_df["day"] / 31)

    return local_df


def add_monthly_lag(df) -> pd.DataFrame:
    local_df = df.copy()
    monthly_data_lag = (
        local_df.groupby(["country", "brand", "year", "month"], observed=True)[
            "monthly"
        ]
        .min()
        .reset_index()
        .sort_values(["country", "brand", "year", "month"])
    )

    monthly_data_lag["monthly_lag"] = (
        monthly_data_lag.groupby(["country", "brand"], observed=True)["monthly"]
        .shift(12)
        .reset_index()["monthly"]
    )

    monthly_data_lag = monthly_data_lag.drop(columns=["monthly"])

    local_df = local_df.merge(
        monthly_data_lag, on=["country", "brand", "year", "month"], how="left"
    )

    return local_df


def add_phase_bcm_stats(df) -> pd.DataFrame:
    all_phase_bcm_stats = pd.DataFrame()

    for year in pd.unique(df["year"]):
        date_threshold = pd.to_datetime(f"{year}-01-01")

        phase_bcm_stats = (
            df[df["date"] < date_threshold]
            .groupby(["brand", "country", "month"], observed=True)["phase"]
            .agg(["mean", "std", "max", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "phase_bcm_mean",
                    "std": "phase_bcm_std",
                    "max": "phase_bcm_max",
                    "median": "phase_bcm_median",
                    "count": "phase_bcm_count",
                }
            )
        )

        phase_bcm_stats["year"] = year

        all_phase_bcm_stats = pd.concat([all_phase_bcm_stats, phase_bcm_stats], axis=0)

    return df.merge(
        all_phase_bcm_stats, on=["brand", "country", "month", "year"], how="left"
    )


def add_phase_brand_month_stats(df) -> pd.DataFrame:
    all_phase_brand_month_stats = pd.DataFrame()

    for year in pd.unique(df["year"]):
        date_threshold = pd.to_datetime(f"{year}-01-01")

        phase_brand_month_stats = (
            df[df["date"] < date_threshold]
            .groupby(["brand", "month"], observed=True)["phase"]
            .agg(["mean", "std", "max", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "phase_brand_month_mean",
                    "std": "phase_brand_month_std",
                    "max": "phase_brand_month_max",
                    "median": "phase_brand_month_median",
                    "count": "phase_brand_month_count",
                }
            )
        )

        phase_brand_month_stats["year"] = year

        all_phase_brand_month_stats = pd.concat(
            [all_phase_brand_month_stats, phase_brand_month_stats], axis=0
        )

    return df.merge(
        all_phase_brand_month_stats, on=["brand", "month", "year"], how="left"
    )


def add_phase_country_month_stats(df) -> pd.DataFrame:
    all_phase_country_month_stats = pd.DataFrame()

    for year in pd.unique(df["year"]):
        date_threshold = pd.to_datetime(f"{year}-01-01")

        phase_country_month_stats = (
            df[df["date"] < date_threshold]
            .groupby(["country", "month"], observed=True)["phase"]
            .agg(["mean", "std", "max", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "phase_country_month_mean",
                    "std": "phase_country_month_std",
                    "max": "phase_country_month_max",
                    "median": "phase_country_month_median",
                    "count": "phase_country_month_count",
                }
            )
        )

        phase_country_month_stats["year"] = year

        all_phase_country_month_stats = pd.concat(
            [all_phase_country_month_stats, phase_country_month_stats], axis=0
        )

    return df.merge(
        all_phase_country_month_stats, on=["country", "month", "year"], how="left"
    )


def add_phase_country_stats(df) -> pd.DataFrame:
    all_phase_country_stats = pd.DataFrame()

    for year in pd.unique(df["year"]):
        date_threshold = pd.to_datetime(f"{year}-01-01")

        phase_country_stats = (
            df[df["date"] < date_threshold]
            .groupby(["country"], observed=True)["phase"]
            .agg(["mean", "std", "max", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "phase_country_mean",
                    "std": "phase_country_std",
                    "max": "phase_country_max",
                    "median": "phase_country_median",
                    "count": "phase_country_count",
                }
            )
        )

        phase_country_stats["year"] = year

        all_phase_country_stats = pd.concat(
            [all_phase_country_stats, phase_country_stats], axis=0
        )

    return df.merge(all_phase_country_stats, on=["country", "year"], how="left")


def add_phase_brand_stats(df) -> pd.DataFrame:
    all_phase_brand_stats = pd.DataFrame()

    for year in pd.unique(df["year"]):
        date_threshold = pd.to_datetime(f"{year}-01-01")

        phase_brand_stats = (
            df[df["date"] < date_threshold]
            .groupby(["brand"], observed=True)["phase"]
            .agg(["mean", "std", "max", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "phase_brand_mean",
                    "std": "phase_brand_std",
                    "max": "phase_brand_max",
                    "median": "phase_brand_median",
                    "count": "phase_brand_count",
                }
            )
        )

        phase_brand_stats["year"] = year

        all_phase_brand_stats = pd.concat(
            [all_phase_brand_stats, phase_brand_stats], axis=0
        )

    return df.merge(all_phase_brand_stats, on=["brand", "year"], how="left")


def add_phase_dayweek_stats(df) -> pd.DataFrame:
    all_phase_dayweek_stats = pd.DataFrame()

    for year in pd.unique(df["year"]):
        date_threshold = pd.to_datetime(f"{year}-01-01")

        phase_dayweek_stats = (
            df[df["date"] < date_threshold]
            .groupby(["dayweek"], observed=True)["phase"]
            .agg(["mean", "std", "max", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "phase_dayweek_mean",
                    "std": "phase_dayweek_std",
                    "max": "phase_dayweek_max",
                    "median": "phase_dayweek_median",
                    "count": "phase_dayweek_count",
                }
            )
        )

        phase_dayweek_stats["year"] = year

        all_phase_dayweek_stats = pd.concat(
            [all_phase_dayweek_stats, phase_dayweek_stats], axis=0
        )

    return df.merge(all_phase_dayweek_stats, on=["dayweek", "year"], how="left")


def add_phase_wd_stats(df) -> pd.DataFrame:
    all_phase_wd_stats = pd.DataFrame()

    for year in pd.unique(df["year"]):
        date_threshold = pd.to_datetime(f"{year}-01-01")

        phase_wd_stats = (
            df[df["date"] < date_threshold]
            .groupby(["wd"], observed=True)["phase"]
            .agg(["mean", "std", "max", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "phase_wd_mean",
                    "std": "phase_wd_std",
                    "max": "phase_wd_max",
                    "median": "phase_wd_median",
                    "count": "phase_wd_count",
                }
            )
        )

        phase_wd_stats["year"] = year

        all_phase_wd_stats = pd.concat([all_phase_wd_stats, phase_wd_stats], axis=0)

    return df.merge(all_phase_wd_stats, on=["wd", "year"], how="left")


def add_phase_wd_month_stats(df) -> pd.DataFrame:
    all_phase_wd_month_stats = pd.DataFrame()

    for year in pd.unique(df["year"]):
        date_threshold = pd.to_datetime(f"{year}-01-01")

        phase_wd_month_stats = (
            df[df["date"] < date_threshold]
            .groupby(["wd", "month"], observed=True)["phase"]
            .agg(["mean", "std", "max", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "phase_wd_month_mean",
                    "std": "phase_wd_month_std",
                    "max": "phase_wd_month_max",
                    "median": "phase_wd_month_median",
                    "count": "phase_wd_month_count",
                }
            )
        )

        phase_wd_month_stats["year"] = year

        all_phase_wd_month_stats = pd.concat(
            [all_phase_wd_month_stats, phase_wd_month_stats], axis=0
        )

    return df.merge(all_phase_wd_month_stats, on=["wd", "month", "year"], how="left")


def add_phase_base_stats(df) -> pd.DataFrame:
    """
    Adds phase stats for different groups. It is leaking into the validation set as we're using all data.
    """
    local_df = df[["country", "brand", "month", "dayweek", "wd", "phase"]].copy()

    groups = ["country", "brand", "month", "dayweek", "wd"]

    for g in groups:
        gdf = local_df.groupby(g, observed=True)["phase"]
        name = g if isinstance(g, str) else "_".join(g)
        local_df[f"phase_{name}_mean"] = gdf.transform("mean")
        local_df[f"phase_{name}_max"] = gdf.transform("max")
        local_df[f"phase_{name}_std"] = gdf.transform("std")

    return local_df


def add_gamma_features(df) -> pd.DataFrame:
    local_df = df.copy()

    gamma_fit = pd.read_csv("../data/raw/gamma_fit.csv")

    gamma_fit["gamma_fit"] = gamma_fit["gamma_fit"].apply(
        lambda x: [float(i) for i in x[1:-1].split(",")]
    )

    gamma_fit["alpha_gamma"] = gamma_fit["gamma_fit"].apply(lambda x: float(x[0]))
    gamma_fit["beta_gamma"] = gamma_fit["gamma_fit"].apply(lambda x: float(x[1]))
    gamma_fit["mu_gamma"] = gamma_fit["gamma_fit"].apply(lambda x: float(x[2]))
    gamma_fit.drop(columns=["gamma_fit"], inplace=True)

    return local_df.merge(gamma_fit, how="left", on=["brand", "country", "month"])


def add_gamma_log_features(df) -> pd.DataFrame:
    local_df = df.copy()

    gamma_fit = pd.read_csv("../data/raw/gamma_fit_log.csv")

    return local_df.merge(gamma_fit, how="left", on=["brand", "country", "month"])


def add_norm_features(df) -> pd.DataFrame:
    local_df = df.copy()

    norm = pd.read_csv("../data/raw/norm_fit.csv")

    norm["norm_fit"] = norm["norm_fit"].apply(
        lambda x: [float(i) for i in x[1:-1].split(",")]
    )

    norm["alpha_norm_no_log"] = norm["norm_fit"].apply(lambda x: float(x[0]))
    norm["beta_norm_no_log"] = norm["norm_fit"].apply(lambda x: float(x[1]))
    norm.drop(columns=["norm_fit"], inplace=True)

    return local_df.merge(norm, how="left", on=["brand", "country", "month"])


def add_norm_log_features(df) -> pd.DataFrame:
    local_df = df.copy()

    norm_fit = pd.read_csv("../data/raw/norm_fit_log.csv")

    return local_df.merge(norm_fit, how="left", on=["brand", "country", "month"])


def add_simple_categorical_facts(df) -> pd.DataFrame:
    df["country_nunique_brands"] = df.groupby("country", observed=True)[
        "brand"
    ].transform("nunique")

    df["brand_nunique_countries"] = df.groupby("brand", observed=True)[
        "country"
    ].transform("nunique")

    df["brand_nunique_years"] = df.groupby("brand", observed=True)["year"].transform(
        "nunique"
    )

    temp = (
        df.groupby("country", observed=True)["main_channel"]
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
        .add_prefix("country_main_channel_")
        .reset_index()
    )
    return df.merge(temp, how="left", on="country")


def fix_day_week_count(df) -> pd.DataFrame:
    temp = df[["year", "month", "day", "dayweek"]].drop_duplicates()

    # count the number of dayweeks in each month
    temp = temp.groupby(["year", "month", "dayweek"]).size().reset_index(name="counts")

    temp = (
        temp.pivot(index=["year", "month"], columns="dayweek", values="counts")
        .reset_index()
        .fillna(0)
    )
    temp.columns = [
        "year",
        "month",
        "dayweek_0",
        "dayweek_1",
        "dayweek_2",
        "dayweek_3",
        "dayweek_4",
        "dayweek_5",
        "dayweek_6",
    ]

    return df.drop(
        columns=[
            "n_weekday_0",
            "n_weekday_1",
            "n_weekday_2",
            "n_weekday_3",
            "n_weekday_4",
        ]
    ).merge(temp, how="left", on=["year", "month"])


def add_lagging(df, n=3):
    n += 1
    local_df = df.copy()
    # groupby brand and country, for each date, get the phase from the previous year
    temp = local_df[["brand", "country", "year", "month", "day", "phase"]].copy()
    # shift the phase by 1 year, 2 years, 3 years, 5 years
    for i in range(1, n):
        temp[f"year_{i}"] = temp["year"] - i
        temp[f"phase_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month", "day"],
            right_on=["brand", "country", "year", "month", "day"],
            how="left",
        )["phase_y"]
        temp.drop(columns=[f"year_{i}"], inplace=True)
    temp = temp.drop(columns=["phase"])
    local_df = local_df.merge(
        temp, on=["brand", "country", "year", "month", "day"], how="left"
    )

    # same but wd
    temp = local_df[
        ["brand", "country", "year", "month", "wd", "n_nwd_bef", "n_nwd_aft", "phase"]
    ].copy()
    # shift the phase by 1 year, 2 years, 3 years, 5 years
    for i in range(1, n):
        temp[f"year_{i}"] = temp["year"] - i
        # Merge 'temp' DataFrame with itself based on specified columns
        a = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month", "wd"],
            right_on=["brand", "country", "year", "month", "wd"],
            how="left",
        )[["n_nwd_bef_y", "n_nwd_aft_y", "phase_y"]]
        # Rename columns in 'a' DataFrame
        a.columns = [f"n_nwd_bef_{i}", f"n_nwd_aft_{i}", f"phase_wd_{i}"]
        # Merge 'temp' DataFrame with 'a' DataFrame using indexes
        temp = temp.merge(a, left_index=True, right_index=True, how="left")
        temp.drop(
            columns=[f"year_{i}"],
            inplace=True,
        )
    temp = temp.drop(columns=["phase", "n_nwd_bef", "n_nwd_aft"])
    local_df = local_df.merge(
        temp, on=["brand", "country", "year", "month", "wd"], how="left"
    )

    # same but monthly
    temp = local_df[["brand", "country", "year", "month", "day", "phase"]]
    # group by brand, country, year, month, and get the mean, median, std, min, max of phase
    temp = (
        temp.groupby(["brand", "country", "year", "month"], observed=True)["phase"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    temp.columns = [
        "brand",
        "country",
        "year",
        "month",
        "phase_month_mean",
        "phase_month_median",
        "phase_month_std",
        "phase_month_min",
        "phase_month_max",
    ]
    # shift the phase by 1 year, 2 years, 3 years, 5 years
    for i in range(1, n):
        temp[f"year_{i}"] = temp["year"] - i
        temp[f"phase_month_mean_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month"],
            right_on=["brand", "country", "year", "month"],
            how="left",
        )["phase_month_mean_y"]
        temp[f"phase_month_median_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month"],
            right_on=["brand", "country", "year", "month"],
            how="left",
        )["phase_month_median_y"]
        temp[f"phase_month_std_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month"],
            right_on=["brand", "country", "year", "month"],
            how="left",
        )["phase_month_std_y"]
        temp[f"phase_month_min_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month"],
            right_on=["brand", "country", "year", "month"],
            how="left",
        )["phase_month_min_y"]
        temp[f"phase_month_max_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month"],
            right_on=["brand", "country", "year", "month"],
            how="left",
        )["phase_month_max_y"]
        temp.drop(columns=[f"year_{i}"], inplace=True)
    temp = temp.drop(
        columns=[
            "phase_month_mean",
            "phase_month_median",
            "phase_month_std",
            "phase_month_min",
            "phase_month_max",
        ]
    )
    local_df = local_df.merge(
        temp, on=["brand", "country", "year", "month"], how="left"
    )

    # same but week
    temp = local_df[["brand", "country", "year", "month", "week", "phase"]]
    # group by brand, country, year, month, and get the mean, median, std, min, max of phase
    temp = (
        temp.groupby(["brand", "country", "year", "month", "week"], observed=True)[
            "phase"
        ]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    temp.columns = [
        "brand",
        "country",
        "year",
        "month",
        "week",
        "phase_week_mean",
        "phase_week_median",
        "phase_week_std",
        "phase_week_min",
        "phase_week_max",
    ]
    # shift the phase by 1 year, 2 years, 3 years, 5 years
    for i in range(1, n):
        temp[f"year_{i}"] = temp["year"] - i
        temp[f"phase_week_mean_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month", "week"],
            right_on=["brand", "country", "year", "month", "week"],
            how="left",
        )["phase_week_mean_y"]
        temp[f"phase_week_median_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month", "week"],
            right_on=["brand", "country", "year", "month", "week"],
            how="left",
        )["phase_week_median_y"]
        temp[f"phase_week_std_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month", "week"],
            right_on=["brand", "country", "year", "month", "week"],
            how="left",
        )["phase_week_std_y"]
        temp[f"phase_week_min_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month", "week"],
            right_on=["brand", "country", "year", "month", "week"],
            how="left",
        )["phase_week_min_y"]
        temp[f"phase_week_max_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month", "week"],
            right_on=["brand", "country", "year", "month", "week"],
            how="left",
        )["phase_week_max_y"]
        temp.drop(columns=[f"year_{i}"], inplace=True)
    temp = temp.drop(
        columns=[
            "phase_week_mean",
            "phase_week_median",
            "phase_week_std",
            "phase_week_min",
            "phase_week_max",
        ]
    )
    local_df = local_df.merge(
        temp, on=["brand", "country", "year", "month", "week"], how="left"
    )

    # calculate the diference between phase_wd and the phase_month_mean
    for i in range(1, n):
        local_df[f"phase_wd_mean_diff_{i}"] = abs(
            local_df[f"phase_wd_{i}"] - local_df[f"phase_month_mean_{i}"]
        )
        local_df[f"phase_wd_median_diff_{i}"] = abs(
            local_df[f"phase_wd_{i}"] - local_df[f"phase_month_median_{i}"]
        )

        local_df[f"phase_day_mean_diff_{i}"] = abs(
            local_df[f"phase_{i}"] - local_df[f"phase_month_mean_{i}"]
        )
        local_df[f"phase_day_median_diff_{i}"] = abs(
            local_df[f"phase_{i}"] - local_df[f"phase_month_median_{i}"]
        )

        # week
        local_df[f"phase_wd_mean_diff_week_{i}"] = abs(
            local_df[f"phase_wd_{i}"] - local_df[f"phase_week_mean_{i}"]
        )
        local_df[f"phase_wd_median_diff_week_{i}"] = abs(
            local_df[f"phase_wd_{i}"] - local_df[f"phase_week_median_{i}"]
        )

        local_df[f"phase_day_mean_diff_week_{i}"] = abs(
            local_df[f"phase_{i}"] - local_df[f"phase_week_mean_{i}"]
        )
        local_df[f"phase_day_median_diff_week_{i}"] = abs(
            local_df[f"phase_{i}"] - local_df[f"phase_week_median_{i}"]
        )

    return local_df


def add_phase_0_lag_stats(df, n=3):
    n += 1
    local_df = df.copy()

    temp = local_df[["brand", "country", "year", "month", "wd", "phase"]].copy()
    # if it is 0 phase is 1
    temp["phase_is_0"] = np.where(temp["phase"] == 0, 1, 0)
    # shift the phase by 1 year, 2 years, 3 years, 5 years
    for i in range(1, n):
        temp[f"year_{i}"] = temp["year"] - i
        temp[f"phase_is_0_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month", "wd"],
            right_on=["brand", "country", "year", "month", "wd"],
            how="left",
        )["phase_is_0_y"]
        temp.drop(columns=[f"year_{i}"], inplace=True)
    temp = temp.drop(columns=["phase", "phase_is_0"])
    local_df = local_df.merge(
        temp, on=["brand", "country", "year", "month", "wd"], how="left"
    )
    # same but monthly
    temp = local_df[["brand", "country", "year", "month", "day", "phase"]]
    # group by brand, country, year, month, and get the count of phase == 0
    temp = (
        temp.groupby(["brand", "country", "year", "month"], observed=True)["phase"]
        .agg(lambda x: (x == 0).sum())
        .reset_index()
    )
    temp.columns = [
        "brand",
        "country",
        "year",
        "month",
        "phase_month_is_0",
    ]
    # shift the phase by 1 year, 2 years, 3 years, 5 years
    for i in range(1, n):
        temp[f"year_{i}"] = temp["year"] - i
        temp[f"phase_month_is_0_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month"],
            right_on=["brand", "country", "year", "month"],
            how="left",
        )["phase_month_is_0_y"]
        temp.drop(columns=[f"year_{i}"], inplace=True)
    temp = temp.drop(
        columns=[
            "phase_month_is_0",
        ]
    )
    local_df = local_df.merge(
        temp, on=["brand", "country", "year", "month"], how="left"
    )

    # same but week
    temp = local_df[["brand", "country", "year", "month", "week", "phase"]]
    # group by brand, country, year, month, and get the count of phase == 0
    temp = (
        temp.groupby(["brand", "country", "year", "month", "week"], observed=True)[
            "phase"
        ]
        .agg(lambda x: (x == 0).sum())
        .reset_index()
    )
    temp.columns = [
        "brand",
        "country",
        "year",
        "month",
        "week",
        "phase_week_is_0",
    ]
    # shift the phase by 1 year, 2 years, 3 years, 5 years
    for i in range(1, n):
        temp[f"year_{i}"] = temp["year"] - i
        temp[f"phase_week_is_0_{i}"] = temp.merge(
            temp,
            left_on=["brand", "country", f"year_{i}", "month", "week"],
            right_on=["brand", "country", "year", "month", "week"],
            how="left",
        )["phase_week_is_0_y"]
        temp.drop(columns=[f"year_{i}"], inplace=True)
    temp = temp.drop(
        columns=[
            "phase_week_is_0",
        ]
    )
    local_df = local_df.merge(
        temp, on=["brand", "country", "year", "month", "week"], how="left"
    )

    return local_df


def add_sample_weight(df):
    local_df = df.copy()

    local_df["sample_weight"] = local_df["year"].clip(0, 2021)
    local_df["sample_weight"] = (
        local_df["sample_weight"] - local_df["sample_weight"].min() + 1
    )
    local_df["sample_weight"] = (
        local_df["sample_weight"] / local_df["sample_weight"].max()
    )
    return local_df
