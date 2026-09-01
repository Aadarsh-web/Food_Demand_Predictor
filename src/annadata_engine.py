import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb

# =========================================================
# LOAD ALL REQUIRED DATA
# =========================================================

model_package = joblib.load(
    "../models/food_demand_xgb.pkl"
)

model = model_package["model"]
model_features = model_package["features"]

recipes = pd.read_csv("../data/recipes.csv")
prices = pd.read_csv("../data/ingredient_prices.csv")
ngos = pd.read_csv("../data/ngos.csv")

# Load food-specific safety rules from the AnnaData Excel file.
safety_data = pd.read_excel("../data/ANANDATA.xlsx")

# Clean column names.
safety_data.columns = safety_data.columns.str.strip()

# Remove blank/category rows.
safety_data = safety_data[
    safety_data["NAME OF FOOD ITEM"].notna()
]

# Keep only actual food items.
category_rows = [
    "COOKED FOOD",
    "BEVERAGES",
    "DAIRY/PERISHABLE",
    "FRESH/RAW",
    "CHUTNEYS",
    "BAKERY/DRY/PACKAGED"
]

safety_data = safety_data[
    ~safety_data["NAME OF FOOD ITEM"].isin(category_rows)
]

# Make food names consistent with our menu names.
safety_data["NAME OF FOOD ITEM"] = (
    safety_data["NAME OF FOOD ITEM"]
    .str.strip()
    .str.title()
    .str.replace(r"\s+", " ",regex=True)
)


# =========================================================
# 1. DEMAND PREDICTION
# =========================================================

def predict_demand(
    attendance,
    meal_type,
    menu,
    rain,
    temperature,
    holiday,
    exam,
    campus_event,
    previous_consumption,
    previous_surplus,
    day_of_week,
    month,
    season
):
    """
    Predict food demand using the trained XGBoost model.
    """

    input_data = pd.DataFrame([{
        "day_of_week": day_of_week,
        "month": month,
        "season": season,
        "attendance": attendance,
        "meal_type": meal_type,
        "menu": menu,
        "rain": rain,
        "temperature": temperature,
        "holiday": holiday,
        "exam": exam,
        "campus_event": campus_event,
        "previous_consumption": previous_consumption,
        "previous_surplus": previous_surplus
    }])

    # Encode categorical variables exactly as during training.
    input_data = pd.get_dummies(
        input_data,
        columns=[
            "season",
            "meal_type",
            "menu"
        ]
    )

    # Make sure prediction columns exactly match
    # the columns used when training XGBoost.
    input_data = input_data.reindex(
        columns=model_features,
        fill_value=0
    )

    prediction_data = xgb.DMatrix(
        input_data
    )

    prediction = model.predict(
        prediction_data
    )[0]

    prediction = max(
        0,
        min(
            prediction,
            attendance
        )
    )

    return {
        "prediction": round(prediction)
    }
# =========================================================
# 2. PREPARATION QUANTITY
# =========================================================

def calculate_preparation(
    predicted_consumption,
    uncertainty,
    attendance
):
    """
    Calculate how much food should be prepared.

    The safety buffer depends on model uncertainty.
    This is currently a prototype rule.
    """

    # Small minimum buffer for normal variation.
    base_buffer = 5

    # Increase the buffer when the model is less certain.
    dynamic_buffer = round(
        base_buffer + uncertainty
    )

    # Recommended preparation.
    recommended = (
        predicted_consumption
        + dynamic_buffer
    )

    # Never recommend more meals than expected attendance.
    recommended = min(
        recommended,
        attendance
    )

    safety_buffer = round(
        recommended
        - predicted_consumption
    )

    return {
        "predicted_consumption": predicted_consumption,
        "uncertainty": uncertainty,
        "safety_buffer": safety_buffer,
        "recommended_preparation": recommended
    }


# =========================================================
# 3. INGREDIENT CALCULATION
# =========================================================

def calculate_ingredients(menu, servings):
    """
    Calculate ingredient quantities using
    per-serving quantities and explicit units.
    """

    menu_recipe = recipes[
        recipes["menu"].str.strip().str.lower()
        == menu.strip().lower()
    ]

    if menu_recipe.empty:
        raise ValueError(
            f"No recipe found for {menu}"
        )

    ingredients = []

    for _, row in menu_recipe.iterrows():

        quantity = (
            float(row["quantity_per_serving"])
            * servings
        )

        ingredients.append({
            "ingredient": row["ingredient"],
            "quantity": round(quantity, 3),
            "unit": row["unit"]
        })

    return ingredients

# =========================================================
# 4. FOOD COST
# =========================================================

def calculate_cost(ingredients):
    """
    Calculate the estimated cost of all ingredients.
    """

    total_cost = 0
    breakdown = []

    for item in ingredients:

        ingredient = item["ingredient"]
        quantity = item["quantity"]

        price_row = prices[
            prices["ingredient"] == ingredient
        ]

        if price_row.empty:
            raise ValueError(
                f"No price found for {ingredient}"
            )

        price = price_row.iloc[0]["price_per_unit"]

        cost = quantity * price

        total_cost += cost

        breakdown.append({
         "ingredient": ingredient,
         "quantity": quantity,
         "unit": item["unit"],
         "unit_price": price,
         "cost": round(cost, 2)
        })

    return {
        "total_cost": round(total_cost, 2),
        "breakdown": breakdown
    }


# =========================================================
# 5. SURPLUS CALCULATION
# =========================================================

def calculate_surplus(
    prepared_quantity,
    consumed_quantity
):
    """
    Calculate the amount of food remaining after the meal.
    """

    surplus = max(
        0,
        prepared_quantity - consumed_quantity
    )

    return surplus


# =========================================================
# 6. FOOD SAFETY GATE
# =========================================================

def check_food_safety(
    menu,
    surplus_quantity,
    storage_temperature,
    storage_type,
    storage_duration_hours
):

    if surplus_quantity <= 0:
        return {
            "status": "NO_SURPLUS",
            "reason": "No surplus food remains."
        }

    # Find food in Excel
    # Normalize the menu name before searching.
    menu_key = " ".join(str(menu).strip().lower().split())

    # Case-insensitive, whitespace-safe lookup.
    food = safety_data[
    safety_data["NAME OF FOOD ITEM"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    == menu_key
    ]

    if food.empty:
        return {
            "status": "REVIEW",
            "reason": f"No safety record found for {menu}."
        }

    food = food.iloc[0]

    condition = str(food["CONDITION"])
    temp_rule = str(food["STORAGE TEMPERATURE"])
    max_time = str(food["MAXIMUM STORAGE TIME"])
    allergens = str(food["ALLERGENS"])
    notes = str(food["NOTES"])

    # -----------------------------
    # TEMPERATURE CHECK
    # -----------------------------

    temperature_ok = True

    if "65" in temp_rule and "ABOVE" in temp_rule.upper():

        if storage_temperature < 65:
            temperature_ok = False

    elif "5" in temp_rule and (
        "BELOW" in temp_rule.upper()
        or "AT 5" in temp_rule.upper()
    ):

        if storage_temperature > 5:
            temperature_ok = False

    # -----------------------------
    # STORAGE TYPE CHECK
    # -----------------------------

    type_ok = True

    if "HOT" in condition.upper():

        if storage_type.lower() != "hot":
            type_ok = False

    elif "COLD" in condition.upper():

        if storage_type.lower() != "cold":
            type_ok = False

    def check_food_safety(
    menu,
    surplus_quantity,
    storage_temperature,
    storage_type,
    storage_duration_hours
):

     if surplus_quantity <= 0:
        return {
            "status": "NO_SURPLUS",
            "reason": "No surplus food remains."
        }

    # Find the food in the Excel safety database.
    menu_key = " ".join(
        str(menu).strip().lower().split()
    )

    food = safety_data[
        safety_data["NAME OF FOOD ITEM"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        == menu_key
    ]

    if food.empty:
        return {
            "status": "REVIEW",
            "reason": f"No safety record found for {menu}."
        }

    food = food.iloc[0]

    # Get the actual safety information from Excel.
    condition = str(
        food["CONDITION"]
    ).strip()

    temperature_rule = str(
        food["STORAGE TEMPERATURE"]
    ).strip()

    maximum_storage_time = str(
        food["MAXIMUM STORAGE TIME"]
    ).strip()

    allergens = str(
        food["ALLERGENS"]
    ).strip()

    instructions = str(
        food["NOTES"]
    ).strip()

    # --------------------------------------------------
    # TEMPERATURE CHECK
    # --------------------------------------------------

    temperature_ok = True

    rule = temperature_rule.upper()

    if "65" in rule and "ABOVE" in rule:
        if storage_temperature < 65:
            temperature_ok = False

    elif "5" in rule and (
        "BELOW" in rule
        or "AT 5" in rule
    ):
        if storage_temperature > 5:
            temperature_ok = False

    # --------------------------------------------------
    # STORAGE TIME CHECK
    # --------------------------------------------------

    time_ok = True
    maximum_hours = None

    time_rule = maximum_storage_time.upper()

    if "2 HOUR" in time_rule:
        maximum_hours = 2

    elif "24 HOUR" in time_rule:
        maximum_hours = 24

    elif "2-3 DAYS" in time_rule:
        maximum_hours = 72

    elif "3-5 DAYS" in time_rule:
        maximum_hours = 120

    elif "1-2 WEEKS" in time_rule:
        maximum_hours = 336

    elif "30-90 DAYS" in time_rule:
        maximum_hours = 2160

    if maximum_hours is not None:
        if storage_duration_hours > maximum_hours:
            time_ok = False

    # --------------------------------------------------
    # STORAGE TYPE CHECK
    # --------------------------------------------------

    storage_ok = True

    if storage_type.lower() == "hot":

        if storage_temperature < 65:
            storage_ok = False

    elif storage_type.lower() == "cold":

        if storage_temperature > 5:
            storage_ok = False

    else:
        return {
            "status": "REVIEW",
            "reason": "Unknown storage type.",
            "food": menu,
            "condition": condition,
            "temperature_rule": temperature_rule,
            "maximum_storage_time": maximum_storage_time,
            "allergens": allergens,
            "instructions": instructions
        }

    # --------------------------------------------------
    # REJECT IF TIME FAILED
    # --------------------------------------------------

    if not time_ok:
        return {
            "status": "REJECT",
            "reason": (
                f"Storage time exceeded. "
                f"Maximum allowed: "
                f"{maximum_hours} hours."
            ),
            "food": menu,
            "condition": condition,
            "temperature_rule": temperature_rule,
            "maximum_storage_time": maximum_storage_time,
            "allergens": allergens,
            "instructions": instructions
        }

    # --------------------------------------------------
    # REJECT IF TEMPERATURE FAILED
    # --------------------------------------------------

    if not temperature_ok:
        return {
            "status": "REJECT",
            "reason": (
                "Required storage temperature "
                "was not maintained."
            ),
            "food": menu,
            "condition": condition,
            "temperature_rule": temperature_rule,
            "maximum_storage_time": maximum_storage_time,
            "allergens": allergens,
            "instructions": instructions
        }

    # --------------------------------------------------
    # REJECT IF STORAGE CONDITION FAILED
    # --------------------------------------------------

    if not storage_ok:
        return {
            "status": "REJECT",
            "reason": (
                "Incorrect storage condition."
            ),
            "food": menu,
            "condition": condition,
            "temperature_rule": temperature_rule,
            "maximum_storage_time": maximum_storage_time,
            "allergens": allergens,
            "instructions": instructions
        }

    # --------------------------------------------------
    # PASSED
    # --------------------------------------------------

    return {
        "status": "ELIGIBLE_FOR_REVIEW",
        "reason": (
            "Food-specific safety checks passed."
        ),
        "food": menu,
        "condition": condition,
        "temperature_rule": temperature_rule,
        "maximum_storage_time": maximum_storage_time,
        "allergens": allergens,
        "instructions": instructions
    }

# =========================================================
# 7. NGO MATCHING
# =========================================================

def match_ngos(
    menu,
    quantity,
    location,
    hostel_latitude,
    hostel_longitude,
    remaining_safe_hours
 ):

    matches = []

    # Approximate travel-time buffer.
    # We use the NGO's estimated pickup time for V1.
    available_minutes = remaining_safe_hours * 60

    for _, ngo in ngos.iterrows():

        # Must currently be accepting pickups.
        if int(ngo["pickup_available"]) != 1:
            continue

        # Must have enough capacity.
        if ngo["max_servings"] <= 0:
            continue

        # Must accept this food.
        accepted_food = str(
            ngo["food_types"]
        ).strip().lower()

        menu_name = menu.strip().lower()

        accepts_food = (
            accepted_food == "any"
            or menu_name in accepted_food
            or accepted_food in menu_name
        )

        if not accepts_food:
            continue

        # Pickup must happen before food safety window expires.
        pickup_minutes = float(
            ngo["avg_pickup_minutes"]
        )

        if pickup_minutes > available_minutes:
            continue

        # Calculate approximate geographic distance.
        lat_difference = (
            float(ngo["latitude"])
            - hostel_latitude
        )

        lon_difference = (
            float(ngo["longitude"])
            - hostel_longitude
        )

        distance = (
            (lat_difference ** 2)
            + (lon_difference ** 2)
        ) ** 0.5

        # Actual quantity given to this NGO.
        allocated_quantity = min(
            quantity,
            int(ngo["max_servings"])
        )

        matches.append({
            "ngo": ngo["ngo_name"],
            "distance_score": round(distance, 4),
            "pickup_minutes": pickup_minutes,
            "remaining_safe_hours": remaining_safe_hours,
            "allocated_servings": allocated_quantity
        })

    # Closest/fastest eligible NGO first.
    matches.sort(
        key=lambda x: (
            x["pickup_minutes"],
            x["distance_score"]
        )
    )

    return matches


# =========================================================
# 8. COMPLETE ANNADATA WORKFLOW
# =========================================================

def run_annadata(
    attendance,
    meal_type,
    menu,
    rain,
    temperature,
    holiday,
    exam,
    campus_event,
    previous_consumption,
    previous_surplus,
    day_of_week,
    month,
    season,
    actual_consumed,
    storage_temperature,
    storage_type,
    storage_duration_hours,
    hostel_latitude,
    hostel_longitude,
    location
):

    # =====================================================
    # DEMAND PREDICTION
    # =====================================================

    prediction_result = predict_demand(
        attendance,
        meal_type,
        menu,
        rain,
        temperature,
        holiday,
        exam,
        campus_event,
        previous_consumption,
        previous_surplus,
        day_of_week,
        month,
        season
    )

    predicted_consumption = prediction_result["prediction"]


    # =====================================================
    # PREPARATION
    # =====================================================

    # Prototype uncertainty.
    uncertainty = 19

    preparation = calculate_preparation(
        predicted_consumption,
        uncertainty,
        attendance
    )


    # =====================================================
    # INGREDIENTS
    # =====================================================

    ingredients = calculate_ingredients(
        menu,
        preparation["recommended_preparation"]
    )


    # =====================================================
    # COST
    # =====================================================

    cost = calculate_cost(
        ingredients
    )


    # =====================================================
    # SURPLUS
    # =====================================================

    surplus = calculate_surplus(
        preparation["recommended_preparation"],
        actual_consumed
    )


    # =====================================================
    # SAFETY
    # =====================================================

    safety = check_food_safety(
        menu,
        surplus,
        storage_temperature,
        storage_type,
        storage_duration_hours
    )


    # =====================================================
    # NGO MATCHING
    # =====================================================

    ngo_matches = []

    if (
        surplus > 0
        and safety["status"] == "ELIGIBLE_FOR_REVIEW"
    ):

        # Extract remaining safe time from the Excel rule.
        remaining_safe_hours = 0

        maximum_storage_time = safety.get(
            "maximum_storage_time",
            ""
        )

        time_rule = str(
            maximum_storage_time
        ).upper()

        if "2 HOUR" in time_rule:
            maximum_hours = 2

        elif "24 HOUR" in time_rule:
            maximum_hours = 24

        elif "2-3 DAYS" in time_rule:
            maximum_hours = 72

        elif "3-5 DAYS" in time_rule:
            maximum_hours = 120

        elif "1-2 WEEKS" in time_rule:
            maximum_hours = 336

        elif "30-90 DAYS" in time_rule:
            maximum_hours = 2160

        else:
            maximum_hours = 0


        remaining_safe_hours = max(
            0,
            maximum_hours - storage_duration_hours
        )


        if remaining_safe_hours > 0:

            ngo_matches = match_ngos(
                menu,
                surplus,
                location,
                hostel_latitude,
                hostel_longitude,
                remaining_safe_hours
            )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {
        "prediction": predicted_consumption,
        "preparation": preparation,
        "ingredients": ingredients,
        "cost": cost,
        "actual_consumed": actual_consumed,
        "surplus": surplus,
        "safety": safety,
        "ngo_matches": ngo_matches
    }

# =========================================================
# RUN ANNADATA USING DYNAMIC INPUT
# =========================================================

if __name__ == "__main__":

    # Load today's meal information.
    with open("../data/today_meal.json", "r") as file:
        meal_data = json.load(file)

    # Run the complete AnnaData engine.
    result = run_annadata(**meal_data)

    print("\n")
    print("========================================")
    print("          ANNADATA AI ENGINE")
    print("========================================")

    print("\n--- DEMAND ---")

    print(
        "Predicted consumption:",
        result["prediction"]
    )

    print(
        "Model uncertainty:",
        result["preparation"]["uncertainty"]
    )

    print(
        "Safety buffer:",
        result["preparation"]["safety_buffer"]
    )

    print(
        "Recommended preparation:",
        result["preparation"]["recommended_preparation"]
    )

    print("\n--- INGREDIENTS ---")

    for item in result["ingredients"]:

        print(
            f"{item['ingredient']}: "
            f"{item['quantity']}"
            f"{item['unit']}"
        )

    print("\n--- COST ---")

    print(
        "Estimated food cost: ₹",
        result["cost"]["total_cost"]
    )

    print("\n--- AFTER MEAL ---")

    print(
        "Actual consumed:",
        result["actual_consumed"]
    )

    print(
        "Surplus:",
        result["surplus"]
    )

    print("\n--- SAFETY ---")

    print(
        "Status:",
        result["safety"]["status"]
    )

    print(
        "Reason:",
        result["safety"]["reason"]
    )

    if "food" in result["safety"]:

        print(
            "Food:",
            result["safety"]["food"]
        )

        print(
            "Condition:",
            result["safety"]["condition"]
        )

        print(
            "Temperature Rule:",
            result["safety"]["temperature_rule"]
        )

        print(
            "Maximum Storage Time:",
            result["safety"]["maximum_storage_time"]
        )

        print(
            "Allergens:",
            result["safety"]["allergens"]
        )

        print(
            "Instructions:",
            result["safety"]["instructions"]
        )

    print("\n--- NGO MATCHING ---")

    if result["ngo_matches"]:

        for ngo in result["ngo_matches"]:

            print(
                f"{ngo['ngo']} → "
                f"{ngo['allocated_servings']} servings "
                f"({ngo['pickup_minutes']} min pickup)"
            )

    else:

        print("No NGO allocation available.")