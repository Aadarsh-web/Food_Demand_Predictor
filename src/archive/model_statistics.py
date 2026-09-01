import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
from xgboost import XGBRegressor


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
    columns=["season", "meal_type", "menu"]
)


# =========================================================
# TIME-BASED SPLIT
# =========================================================

split_date = pd.Timestamp("2025-01-01")

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model_package = joblib.load(
    "../models/food_demand_xgb.pkl"
)

model = model_package["model"]


# =========================================================
# PREDICTIONS
# =========================================================

dtest = xgb.DMatrix(X_test)

predictions = model.predict(dtest)

predictions = pd.Series(
    predictions,
    index=y_test.index
)

errors = predictions - y_test
absolute_errors = errors.abs()


# =========================================================
# STATISTICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

maximum_error = absolute_errors.max()

maximum_underprediction = errors.min()

maximum_overprediction = errors.max()

underpredictions = (errors < 0).sum()

overpredictions = (errors > 0).sum()

exact_predictions = (errors == 0).sum()

total_predictions = len(errors)

underprediction_rate = (
    underpredictions / total_predictions
) * 100

overprediction_rate = (
    overpredictions / total_predictions
) * 100

exact_prediction_rate = (
    exact_predictions / total_predictions
) * 100

average_error = errors.mean()


# =========================================================
# DISPLAY
# =========================================================

print("\n========================================")
print("       ANNADATA MODEL STATISTICS")
print("========================================")

print(
    f"\nTest records: {total_predictions}"
)

print(
    f"MAE: {mae:.2f} meals"
)

print(
    f"Average error: {average_error:.2f} meals"
)

print(
    f"Maximum absolute error: "
    f"{maximum_error:.0f} meals"
)

print(
    f"Worst underprediction: "
    f"{maximum_underprediction:.0f} meals"
)

print(
    f"Worst overprediction: "
    f"+{maximum_overprediction:.0f} meals"
)

print("\n--- ERROR DIRECTION ---")

print(
    f"Underpredictions: "
    f"{underpredictions} "
    f"({underprediction_rate:.2f}%)"
)

print(
    f"Overpredictions: "
    f"{overpredictions} "
    f"({overprediction_rate:.2f}%)"
)

print(
    f"Exact predictions: "
    f"{exact_predictions} "
    f"({exact_prediction_rate:.2f}%)"
)