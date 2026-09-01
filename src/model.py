import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import mean_absolute_error

import xgboost as xgb


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("../data/meal_data.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# =========================================================
# FEATURES
# =========================================================

features = [
    "day_of_week",
    "month",
    "season",
    "attendance",
    "meal_type",
    "menu",
    "rain",
    "temperature",
    "holiday",
    "exam",
    "campus_event",
    "previous_consumption",
    "previous_surplus"
]

target = "meals_consumed"


X = df[features].copy()
y = df[target]


# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

X = pd.get_dummies(
    X,
    columns=[
        "season",
        "meal_type",
        "menu"
    ]
)


# =========================================================
# TIME-BASED TRAIN / TEST SPLIT
# =========================================================

split_date = pd.Timestamp("2025-01-01")

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]


# =========================================================
# XGBOOST 95% QUANTILE MODEL
# =========================================================

model_params = {
    "objective": "reg:quantileerror",
    "quantile_alpha": 0.95,

    "max_depth": 6,
    "learning_rate": 0.05,

    "subsample": 0.85,
    "colsample_bytree": 0.85,

    "seed": 42
}


# =========================================================
# XGBOOST DATA MATRICES
# =========================================================

dtrain = xgb.DMatrix(
    X_train,
    label=y_train
)

dtest = xgb.DMatrix(
    X_test,
    label=y_test
)


# =========================================================
# TRAIN
# =========================================================

model = xgb.train(
    params=model_params,
    dtrain=dtrain,
    num_boost_round=500
)


# =========================================================
# EVALUATE
# =========================================================

predictions = model.predict(
    dtest
)

mae = mean_absolute_error(
    y_test,
    predictions
)


print("\n========================================")
print("        ANNADATA XGBOOST MODEL")
print("========================================")

print(
    f"\nTraining records: {len(X_train)}"
)

print(
    f"Testing records: {len(X_test)}"
)

print(
    f"Features used: {len(X.columns)}"
)

print(
    f"Quantile: 95%"
)

print(
    f"MAE: {mae:.2f} meals"
)


# =========================================================
# SAVE MODEL + FEATURE COLUMNS
# =========================================================

model_package = {
    "model": model,
    "features": list(X.columns),
    "model_type": "xgboost_quantile",
    "quantile": 0.95
}


joblib.dump(
    model_package,
    "../models/food_demand_xgb.pkl"
)


print(
    "\nModel saved to "
    "../models/food_demand_xgb.pkl"
)