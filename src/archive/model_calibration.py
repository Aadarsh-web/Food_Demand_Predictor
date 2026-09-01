import pandas as pd
import numpy as np
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

X = df[features].copy()
y = df["meals_consumed"]

X = pd.get_dummies(
    X,
    columns=["season", "meal_type", "menu"]
)


# =========================================================
# TIME SPLIT
# =========================================================

split_date = pd.Timestamp("2025-01-01")

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]


# =========================================================
# TRAIN 95% QUANTILE MODEL
# =========================================================

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:quantileerror",
    quantile_alpha=0.95,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
actual = y_test.to_numpy()


# =========================================================
# CALIBRATION EXPERIMENT
# =========================================================

corrections = [0, 5, 10, 15, 20, 25, 30, 35, 40]

print("\n========================================")
print("       ANNADATA CALIBRATION TEST")
print("========================================")

print(
    "\nCorrection | Under % | Worst Shortage | "
    "Avg Shortage | Avg Surplus | MAE"
)

print("-" * 78)


for correction in corrections:

    calibrated = predictions - correction

    errors = calibrated - actual

    under = errors < 0
    over = errors > 0

    under_count = under.sum()
    over_count = over.sum()

    under_rate = (
        under_count / len(errors)
    ) * 100

    worst_shortage = (
        errors[under].min()
        if under_count > 0
        else 0
    )

    average_shortage = (
        np.abs(errors[under]).mean()
        if under_count > 0
        else 0
    )

    average_surplus = (
        errors[over].mean()
        if over_count > 0
        else 0
    )

    mae = np.abs(errors).mean()

    print(
        f"{correction:>10} | "
        f"{under_rate:>7.2f}% | "
        f"{worst_shortage:>14.0f} | "
        f"{average_shortage:>12.2f} | "
        f"{average_surplus:>11.2f} | "
        f"{mae:>5.2f}"
    )