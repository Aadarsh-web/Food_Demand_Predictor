import pandas as pd
import joblib


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

model = joblib.load("models/food_demand_model.pkl")


# ---------------------------------------------------------
# LOAD RECIPE DATABASE
# ---------------------------------------------------------

recipes = pd.read_csv("data/recipes.csv")


# ---------------------------------------------------------
# DEMAND PREDICTION
# ---------------------------------------------------------

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
    Predict how many meals are likely to be consumed.
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

    prediction = model.predict(input_data)[0]

    # Consumption cannot be negative or greater than attendance.
    prediction = max(0, min(prediction, attendance))

    return round(prediction)


# ---------------------------------------------------------
# INGREDIENT CALCULATOR
# ---------------------------------------------------------

def calculate_ingredients(menu, servings):
    """
    Calculate ingredient quantities required
    for the predicted number of servings.
    """

    menu_recipe = recipes[recipes["menu"] == menu]

    if menu_recipe.empty:
        raise ValueError(f"No recipe found for: {menu}")

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


# ---------------------------------------------------------
# COMPLETE KITCHEN PLAN
# ---------------------------------------------------------

def create_kitchen_plan(
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
    Create a complete preparation plan.

    This combines:
    1. ML demand prediction
    2. Safety buffer
    3. Recommended preparation quantity
    4. Ingredient calculation
    """

    # STEP 1 — Predict demand using ML.
    predicted_consumption = predict_demand(
        attendance=attendance,
        meal_type=meal_type,
        menu=menu,
        rain=rain,
        temperature=temperature,
        holiday=holiday,
        exam=exam,
        campus_event=campus_event,
        previous_consumption=previous_consumption,
        previous_surplus=previous_surplus,
        day_of_week=day_of_week
    )

    # STEP 2 — Add a temporary safety buffer.
    #
    # This is NOT our final method.
    # Later we will calculate this using prediction uncertainty.
    safety_buffer_percent = 0.03

    recommended_preparation = round(
        predicted_consumption
        * (1 + safety_buffer_percent)
    )

    # Never prepare more than the number of people expected.
    recommended_preparation = min(
        recommended_preparation,
        attendance
    )

    safety_buffer = (
        recommended_preparation
        - predicted_consumption
    )

    # STEP 3 — Convert servings into ingredient quantities.
    ingredients = calculate_ingredients(
        menu,
        recommended_preparation
    )

    # STEP 4 — Return everything as one structured result.
    return {
        "predicted_consumption": predicted_consumption,
        "safety_buffer": safety_buffer,
        "recommended_preparation": recommended_preparation,
        "ingredients": ingredients
    }


# ---------------------------------------------------------
# TEST THE COMPLETE ENGINE
# ---------------------------------------------------------

if __name__ == "__main__":

    kitchen_plan = create_kitchen_plan(
        attendance=600,
        meal_type="Lunch",
        menu="Chicken Rice",
        rain=1,
        temperature=28,
        holiday=0,
        exam=0,
        campus_event=0,
        previous_consumption=570,
        previous_surplus=30,
        day_of_week=2
    )

    print("\n================================")
    print("       ANNADATA KITCHEN PLAN")
    print("================================")

    print(
        "\nPredicted consumption:",
        kitchen_plan["predicted_consumption"]
    )

    print(
        "Safety buffer:",
        kitchen_plan["safety_buffer"]
    )

    print(
        "Recommended preparation:",
        kitchen_plan["recommended_preparation"]
    )

    print("\nIngredients:")

    for item in kitchen_plan["ingredients"]:

        print(
            f"{item['ingredient']}: "
            f"{item['quantity']}"
        )