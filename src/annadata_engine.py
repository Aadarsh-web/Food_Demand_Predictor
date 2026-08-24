import pandas as pd
import numpy as np
import joblib
import json


# =========================================================
# LOAD ALL REQUIRED DATA
# =========================================================

model = joblib.load("models/food_demand_xgb.pkl")

recipes = pd.read_csv("data/recipes.csv")
prices = pd.read_csv("data/ingredient_prices.csv")
ngos = pd.read_csv("data/ngos.csv")


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
    day_of_week
):
    """
    Predict meal demand using the trained XGBoost model.
    """

    input_data = pd.DataFrame([{
        "day_of_week": day_of_week,
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

    # Ask XGBoost for its prediction.
    prediction = model.predict(input_data)[0]

    # Keep prediction within realistic limits.
    prediction = max(
        0,
        min(prediction, attendance)
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

    safety_buffer = (
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
    Calculate ingredient quantities for the required servings.
    """

    menu_recipe = recipes[
        recipes["menu"] == menu
    ]

    if menu_recipe.empty:
        raise ValueError(
            f"No recipe found for {menu}"
        )

    ingredients = []

    for _, row in menu_recipe.iterrows():

        quantity = (
            row["quantity_per_100"]
            * servings
            / 100
        )

        ingredients.append({
            "ingredient": row["ingredient"],
            "quantity": round(quantity, 2)
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
    surplus_quantity,
    storage_temperature,
    storage_type
):
    """
    Perform the prototype safety checks.

    This does NOT certify food as safe.
    It only determines whether the configured
    prototype checks have passed.
    """

    if surplus_quantity <= 0:

        return {
            "status": "NO_SURPLUS",
            "reason": "No surplus food remains."
        }

    # Prototype thresholds.
    HOT_MIN_TEMP = 65
    COLD_MAX_TEMP = 5

    if storage_type == "hot":

        if storage_temperature < HOT_MIN_TEMP:

            return {
                "status": "REJECT",
                "reason": (
                    "Hot food temperature is below "
                    "the configured threshold."
                )
            }

    elif storage_type == "cold":

        if storage_temperature > COLD_MAX_TEMP:

            return {
                "status": "REJECT",
                "reason": (
                    "Cold food temperature is above "
                    "the configured threshold."
                )
            }

    else:

        return {
            "status": "REVIEW",
            "reason": "Unknown storage type."
        }

    return {
        "status": "ELIGIBLE_FOR_REVIEW",
        "reason": "Configured temperature check passed."
    }


# =========================================================
# 7. NGO MATCHING
# =========================================================

def match_ngos(
    menu,
    quantity,
    location
):
    """
    Find NGOs that can potentially receive the surplus.
    """

    matches = []

    for _, ngo in ngos.iterrows():

        if ngo["location"] != location:
            continue

        if ngo["pickup_available"] != 1:
            continue

        accepts_food = (
            ngo["food_types"] == "Any"
            or menu in ngo["food_types"]
            or menu.split()[0] in ngo["food_types"]
        )

        if not accepts_food:
            continue

        allocation = min(
            quantity,
            ngo["max_servings"]
        )

        matches.append({
            "ngo_name": ngo["ngo_name"],
            "allocation": allocation
        })

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
    actual_consumed,
    storage_temperature,
    storage_type,
    location
):

    # -----------------------------------------------------
    # DEMAND
    # -----------------------------------------------------

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
        day_of_week
    )

    predicted = prediction_result["prediction"]

    #Uncertainty coming soon
    uncertainty = 0

    # -----------------------------------------------------
    # PREPARATION
    # -----------------------------------------------------

    preparation = calculate_preparation(
        predicted,
        uncertainty,
        attendance
    )

    recommended = preparation[
        "recommended_preparation"
    ]

    # -----------------------------------------------------
    # INGREDIENTS
    # -----------------------------------------------------

    ingredients = calculate_ingredients(
        menu,
        recommended
    )

    # -----------------------------------------------------
    # COST
    # -----------------------------------------------------

    cost = calculate_cost(
        ingredients
    )

    # -----------------------------------------------------
    # ACTUAL SURPLUS
    # -----------------------------------------------------

    surplus = calculate_surplus(
        recommended,
        actual_consumed
    )

    # -----------------------------------------------------
    # SAFETY
    # -----------------------------------------------------

    safety = check_food_safety(
        surplus,
        storage_temperature,
        storage_type
    )

    # -----------------------------------------------------
    # NGO MATCHING
    # -----------------------------------------------------

    if safety["status"] == "ELIGIBLE_FOR_REVIEW":

        ngo_matches = match_ngos(
            menu,
            surplus,
            location
        )

    else:

        ngo_matches = []

    # -----------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------

    return {
        "prediction": predicted,
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
    with open("data/today_meal.json", "r") as file:
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

    print("\n--- NGO MATCHING ---")

    if result["ngo_matches"]:

        for ngo in result["ngo_matches"]:

            print(
                f"{ngo['ngo_name']} → "
                f"{ngo['allocation']} servings"
            )

    else:

        print("No NGO allocation available.")