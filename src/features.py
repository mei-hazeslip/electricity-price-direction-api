# src/features.py
import numpy as np
import pandas as pd

def build_features(df):
    """Full improved feature set."""
    df = df.copy()
    X = pd.DataFrame(index=df.index)

    # Time
    hour = df.index.hour
    X["hour"] = hour
    X["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    X["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    X["dayofweek"] = df.index.dayofweek
    X["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    # X["month"] = df.index.month

    # All raw features
    raw_cols = [
        "PV_DayAhead_T", "ProvincialLoad_DayAhead_T", "BiddingSpace_DayAhead_T",
        "InterconnectionPlan_DayAhead_T", "NonMarketUnitOutput_DayAhead_T",
        "Wind_DayAhead_T", "PV_RealTime_T_minus_2", "ProvincialLoad_RealTime_T_minus_2",
        "BiddingSpace_RealTime_T_minus_2", "Interconnection_RealTime_T_minus_2",
        "NonMarketUnitOutput_RealTime_T_minus_2", "Wind_RealTime_T_minus_2",
        "ClearingPrice_RealTime_T_minus_2", "ClearingPrice_DayAhead_T_minus_1",
    ]
    X[raw_cols] = df[raw_cols]

    # Safe 24h ramps (not cross-row .diff())
    for col in ["PV_DayAhead_T", "Wind_DayAhead_T",
                "ProvincialLoad_DayAhead_T", "BiddingSpace_DayAhead_T"]:
        X[f"{col}_ramp_24h"] = df[col] - df[col].shift(24)
        X[f"{col}_ramp_48h"] = df[col] - df[col].shift(48)

    # Price lags (same-hour lookback)
    for lag_days in [1, 2, 3, 7]:
        X[f"price_rt2_lag_{lag_days}d"] = df["ClearingPrice_RealTime_T_minus_2"].shift(lag_days * 24)
        X[f"price_da1_lag_{lag_days}d"] = df["ClearingPrice_DayAhead_T_minus_1"].shift(lag_days * 24)

    # Rolling price mean (3d, 7d) - shifted 1 day to avoid leakage
    price_shifted = df["ClearingPrice_RealTime_T_minus_2"].shift(24)
    X["price_rt2_roll3d"] = price_shifted.rolling(72, min_periods=1).mean()
    X["price_rt2_roll7d"] = price_shifted.rolling(168, min_periods=1).mean()

    # Price gap & context
    X["price_gap_recent"] = df["ClearingPrice_DayAhead_T_minus_1"] - df["ClearingPrice_RealTime_T_minus_2"]

    # Mismatch / forecast error features
    X["pv_diff_da_rt2"] = df["PV_DayAhead_T"] - df["PV_RealTime_T_minus_2"]
    X["wind_diff_da_rt2"] = df["Wind_DayAhead_T"] - df["Wind_RealTime_T_minus_2"]
    X["load_diff_da_rt2"] = df["ProvincialLoad_DayAhead_T"] - df["ProvincialLoad_RealTime_T_minus_2"]
    X["renewable_surplus_da"] = X["pv_diff_da_rt2"] + X["wind_diff_da_rt2"]

    # Residual demand
    X["residual_da"] = (
        df["ProvincialLoad_DayAhead_T"] - df["PV_DayAhead_T"]
        - df["Wind_DayAhead_T"] - df["NonMarketUnitOutput_DayAhead_T"]
    )

    # Hour × feature interactions
    for col in ["BiddingSpace_DayAhead_T", "PV_DayAhead_T",
                "Wind_DayAhead_T", "ProvincialLoad_DayAhead_T"]:
        X[f"{col}_x_sin"] = df[col] * X["sin_hour"]
        X[f"{col}_x_cos"] = df[col] * X["cos_hour"]

    # Missing value handling
    # For lag/ramp/rolling features, it forward-fills, 
    for col in X.columns:
        if "lag_" in col or "roll" in col or "ramp_" in col:
            X[col] = X[col].ffill()
    # then fills remaining missing values with the median.
    X = X.fillna(X.median(numeric_only=True))
    
    return X