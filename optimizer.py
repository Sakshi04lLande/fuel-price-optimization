import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# DATA INGESTION & VALIDATION
# --------------------------------------------------
def load_and_validate_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.drop_duplicates(subset=["date"], inplace=True)

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(method="ffill")

    df = df[(df["price"] > 0) & (df["cost"] > 0) & (df["volume"] >= 0)]
    return df.reset_index(drop=True)

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def create_features(df):
    df = df.copy()

    df["price_vs_comp1"] = df["price"] - df["comp1_price"]
    df["price_vs_comp2"] = df["price"] - df["comp2_price"]
    df["price_vs_comp3"] = df["price"] - df["comp3_price"]

    df["price_lag_1"] = df["price"].shift(1)
    df["volume_lag_1"] = df["volume"].shift(1)

    df["price_ma_7"] = df["price"].rolling(7).mean()
    df["volume_ma_7"] = df["volume"].rolling(7).mean()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df.dropna(inplace=True)
    return df

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
def train_model(df):

    FEATURES = [
        "price", "cost",
        "comp1_price", "comp2_price", "comp3_price",
        "price_vs_comp1", "price_vs_comp2", "price_vs_comp3",
        "price_lag_1", "volume_lag_1",
        "price_ma_7", "volume_ma_7",
        "day_of_week", "month"
    ]

    X = df[FEATURES]
    y = df["volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2 :", r2_score(y_test, preds))

    return model, FEATURES

# --------------------------------------------------
# PRICE OPTIMIZATION
# --------------------------------------------------
def recommend_price(today_data, history_df, model, FEATURES,
                    max_daily_change=0.05, min_margin=0.02):

    if history_df.empty:
        raise ValueError("Feature dataframe is empty after preprocessing.")

    last = history_df.iloc[-1]

    last_price = today_data["price"]
    cost = today_data["cost"]

    low = last_price * (1 - max_daily_change)
    high = last_price * (1 + max_daily_change)
    candidate_prices = np.linspace(low, high, 30)

    best_profit = -np.inf
    best_result = None

    for p in candidate_prices:
        if p <= cost * (1 + min_margin):
            continue

        row = {
            "price": p,
            "cost": cost,
            "comp1_price": today_data["comp1_price"],
            "comp2_price": today_data["comp2_price"],
            "comp3_price": today_data["comp3_price"],
            "price_vs_comp1": p - today_data["comp1_price"],
            "price_vs_comp2": p - today_data["comp2_price"],
            "price_vs_comp3": p - today_data["comp3_price"],
            "price_lag_1": last["price"],
            "volume_lag_1": last["volume"],
            "price_ma_7": last["price_ma_7"],
            "volume_ma_7": last["volume_ma_7"],
            "day_of_week": pd.to_datetime(today_data["date"]).dayofweek,
            "month": pd.to_datetime(today_data["date"]).month
        }

        X_today = pd.DataFrame([row])[FEATURES]
        volume = model.predict(X_today)[0]
        profit = (p - cost) * volume

        if profit > best_profit:
            best_profit = profit
            best_result = (p, volume, profit)

    if best_result is None:
        raise ValueError("No valid price found under margin constraints.")

    return {
    "recommended_price": float(round(best_result[0], 2)),
    "expected_volume": float(round(best_result[1], 1)),
    "expected_profit": float(round(best_result[2], 2))
    }
