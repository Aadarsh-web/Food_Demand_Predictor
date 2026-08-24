import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


DATA_PATH = "data/meal_data.csv"
MODEL_PATH = "models/food_demand_xgb.pkl"


def train_model():

    # Load the latest historical data.
    df = pd.read_csv(DATA_PATH)

    # Separate inputs from the target.
    X = df.drop(
        columns=["meals_consumed", "date"]
    )

    y = df["meals_consumed"]

    # Features containing text.
    categorical_features = [
        "meal_type",
        "menu"
    ]

    # Features containing numbers.
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

    # Convert categorical features into numerical representation.
    preprocessor = ColumnTransformer(
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

    # Create the XGBoost model.
    model = Pipeline([
        (
            "preprocessor",
            preprocessor
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

    # Train on the complete available dataset.
    model.fit(X, y)

    # Save the trained XGBoost model.
    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"XGBoost trained using {len(df)} records."
    )

    print(
        "Model saved to:",
        MODEL_PATH
    )

    return model


if __name__ == "__main__":
    train_model()