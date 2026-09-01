import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


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
# ENCODE
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
# TEST QUANTILES
# =========================================================

quantiles = [0.80, 0.90, 0.95, 0.98]


print("\n========================================")
print("     ANNADATA QUANTILE EXPERIMENT")
print("========================================")

print(
    f"\nTraining records: {len(X_train)}"
)

print(
    f"Testing records: {len(X_test)}"
)


# =========================================================
# TRAIN EACH MODEL
# =========================================================

for quantile in quantiles:

    print(
        f"\n--- Quantile: {quantile:.0%} ---"
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,

        objective="reg:quantileerror",

        quantile_alpha=quantile,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    errors = predictions - y_test.to_numpy()

    absolute_errors = np.abs(errors)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    under_count = np.sum(
        errors < 0
    )

    over_count = np.sum(
        errors > 0
    )

    under_rate = (
        under_count / len(errors)
    ) * 100

    over_rate = (
        over_count / len(errors)
    ) * 100

    worst_under = (
        errors.min()
    )

    worst_over = (
        errors.max()
    )

    average_error = (
        errors.mean()
    )

    average_surplus = (
        errors[errors > 0].mean()
        if over_count > 0
        else 0
    )

    average_shortage = (
        np.abs(errors[errors < 0]).mean()
        if under_count > 0
        else 0
    )

    print(
        f"MAE: {mae:.2f} meals"
    )

    print(
        f"Average error: "
        f"{average_error:.2f} meals"
    )

    print(
        f"Underprediction: "
        f"{under_count} "
        f"({under_rate:.2f}%)"
    )

    print(
        f"Overprediction: "
        f"{over_count} "
        f"({over_rate:.2f}%)"
    )

    print(
        f"Worst underprediction: "
        f"{worst_under:.0f} meals"
    )

    print(
        f"Worst overprediction: "
        f"+{worst_over:.0f} meals"
    )

    print(
        f"Average shortage: "
        f"{average_shortage:.2f} meals"
    )

    print(
        f"Average surplus: "
        f"{average_surplus:.2f} meals"
    )