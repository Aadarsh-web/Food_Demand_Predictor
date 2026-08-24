import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/meal_data.csv")

X = df.drop(columns=["meals_consumed", "date"])
y = df["meals_consumed"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# FEATURES
# =========================================================

categorical_features = [
    "meal_type",
    "menu"
]

numerical_features = [
    "attendance",
    "day_of_week",
    "rain",
    "temperature",
    "holiday",
    "exam",
    "campus_event",
    "previous_consumption",
    "previous_surplus"
]


# =========================================================
# RANDOM FOREST
# =========================================================

rf_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


rf_model = Pipeline([
    (
        "preprocessor",
        rf_preprocessor
    ),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    )
])


print("Training Random Forest...")

rf_model.fit(
    X_train,
    y_train
)


# =========================================================
# RANDOM FOREST EVALUATION
# =========================================================

rf_predictions = rf_model.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)


print(
    "Random Forest MAE:",
    round(rf_mae, 2)
)


# =========================================================
# XGBOOST
# =========================================================

xgb_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


xgb_model = Pipeline([
    (
        "preprocessor",
        xgb_preprocessor
    ),
    (
        "regressor",
        XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=-1
        )
    )
])


print("\nTraining XGBoost...")

xgb_model.fit(
    X_train,
    y_train
)


# =========================================================
# XGBOOST EVALUATION
# =========================================================

xgb_predictions = xgb_model.predict(
    X_test
)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)


print(
    "XGBoost MAE:",
    round(xgb_mae, 2)
)


# =========================================================
# FINAL COMPARISON
# =========================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    "Random Forest MAE:",
    round(rf_mae, 2)
)

print(
    "XGBoost MAE:",
    round(xgb_mae, 2)
)


if xgb_mae < rf_mae:

    print("\nWinner: XGBoost")

elif rf_mae < xgb_mae:

    print("\nWinner: Random Forest")

else:

    print("\nResult: Tie")