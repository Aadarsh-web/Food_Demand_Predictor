import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/meal_data.csv")


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
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# XGBOOST
# =========================================================

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)


# =========================================================
# TRAIN
# =========================================================

model.fit(
    X_train,
    y_train
)


# =========================================================
# EVALUATE
# =========================================================

predictions = model.predict(
    X_test
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
    f"MAE: {mae:.2f} meals"
)


# =========================================================
# SAVE MODEL + FEATURE COLUMNS
# =========================================================

model_package = {
    "model": model,
    "features": list(X.columns)
}

joblib.dump(
    model_package,
    "models/food_demand_xgb.pkl"
)


print(
    "\nModel saved to "
    "models/food_demand_xgb.pkl"
)