import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the historical meal dataset generated in Step 1.
df = pd.read_csv("data/meal_data.csv")


# Compare average consumption for each menu.
menu_consumption = df.groupby("menu")["meals_consumed"].mean()

print("\nAverage consumption by menu:")
print(menu_consumption)

# Compare average consumption on rainy vs non-rainy days.
rain_consumption = df.groupby("rain")["meals_consumed"].mean()

print("\nAverage consumption by rain:")
print(rain_consumption)

# Compare average consumption across different days of the week.
day_consumption = df.groupby("day_of_week")["meals_consumed"].mean()

print("\nAverage consumption by day:")
print(day_consumption)



# Calculate correlations between numerical features and meal consumption.
correlation = df.select_dtypes(include="number").corr()["meals_consumed"].sort_values(ascending=False)

print("\nCorrelation with meals consumed:")
print(correlation)

# Display only correlations with the target, excluding the target itself.
feature_correlation = correlation.drop("meals_consumed")

print("\nFeature correlations:")
print(feature_correlation)

# Check whether any meal has consumption greater than attendance.
invalid_records = df[df["meals_consumed"] > df["attendance"]]

print("\nInvalid consumption records:", len(invalid_records))











from sklearn.model_selection import train_test_split

# Separate the input features (X) from the target we want to predict (y).
X = df.drop(columns=["meals_consumed","date"])
y = df["meals_consumed"]

# Split the dataset into training data and testing data.
# 80% is used for learning and 20% is kept completely unseen for evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



from sklearn.metrics import mean_absolute_error

# Predict the average training-set consumption for every test record.
baseline_prediction = np.full(
    len(y_test),
    y_train.mean()
)

# Measure the baseline's average absolute error.
baseline_mae = mean_absolute_error(y_test, baseline_prediction)

print("\nBaseline MAE:", baseline_mae)


















from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# Separate the input features (X) from the target (y) we want to predict.
X = df.drop(columns=["meals_consumed","date"])
y = df["meals_consumed"]

# Identify categorical and numerical features.
categorical_features = ["meal_type", "menu"]

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

# Convert categorical text into numerical values while leaving numerical features unchanged.
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# Create a pipeline that encodes categorical data and then trains a Random Forest.
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Train the Random Forest using the training data.
model.fit(X_train, y_train)

# Predict consumption for data the model did not see during training.
predictions = model.predict(X_test)

# Measure how far the predictions are from the actual consumption.
model_mae = mean_absolute_error(y_test, predictions)

print("\nBaseline MAE:", baseline_mae)
print("Random Forest MAE:", model_mae)