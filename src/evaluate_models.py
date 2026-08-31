import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/meal_data.csv")

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
# CHRONOLOGICAL SPLIT
# =========================================================

split_date = pd.Timestamp("2025-01-01")

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]


# =========================================================
# TRAIN XGBOOST
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


model.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICT
# =========================================================

predictions = model.predict(
    X_test
)


# =========================================================
# EVALUATE
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)


print("\n========================================")
print("     ANNADATA TIME-BASED EVALUATION")
print("========================================")

print(
    f"\nTraining period: "
    f"{df.loc[train_mask, 'date'].min().date()} "
    f"→ "
    f"{df.loc[train_mask, 'date'].max().date()}"
)

print(
    f"Testing period: "
    f"{df.loc[test_mask, 'date'].min().date()} "
    f"→ "
    f"{df.loc[test_mask, 'date'].max().date()}"
)

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
    f"\nTime-based MAE: {mae:.2f} meals"
)