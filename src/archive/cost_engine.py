import pandas as pd


# ---------------------------------------------------------
# LOAD INGREDIENT PRICES
# ---------------------------------------------------------

prices = pd.read_csv("data/ingredient_prices.csv")


# ---------------------------------------------------------
# CALCULATE COST OF INGREDIENTS
# ---------------------------------------------------------

def calculate_food_cost(ingredients):
    """
    Calculate the total estimated cost of the ingredients
    required for a preparation plan.
    """

    total_cost = 0

    breakdown = []

    for item in ingredients:

        ingredient = item["ingredient"]
        quantity = item["quantity"]

        # Find the price information for this ingredient.
        price_row = prices[
            prices["ingredient"] == ingredient
        ]

        if price_row.empty:
            raise ValueError(
                f"No price found for: {ingredient}"
            )

        price_per_unit = price_row.iloc[0]["price_per_unit"]

        # Calculate cost for this ingredient.
        cost = quantity * price_per_unit

        total_cost += cost

        breakdown.append({
            "ingredient": ingredient,
            "quantity": quantity,
            "unit_price": price_per_unit,
            "cost": round(cost, 2)
        })

    return {
        "total_cost": round(total_cost, 2),
        "breakdown": breakdown
    }


# ---------------------------------------------------------
# ESTIMATE SAVINGS
# ---------------------------------------------------------

def calculate_savings(
    old_servings,
    new_servings,
    cost_per_serving
):
    """
    Estimate how much money could be saved by reducing
    unnecessary preparation.

    This is a simplified prototype calculation.
    """

    old_cost = old_servings * cost_per_serving

    new_cost = new_servings * cost_per_serving

    savings = old_cost - new_cost

    return {
        "old_cost": round(old_cost, 2),
        "new_cost": round(new_cost, 2),
        "estimated_savings": round(savings, 2)
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    test_ingredients = [
        {
            "ingredient": "Rice",
            "quantity": 44
        },
        {
            "ingredient": "Chicken",
            "quantity": 26
        },
        {
            "ingredient": "Onion",
            "quantity": 9
        },
        {
            "ingredient": "Tomato",
            "quantity": 7
        },
        {
            "ingredient": "Oil",
            "quantity": 5
        }
    ]

    result = calculate_food_cost(test_ingredients)

    print("\n==============================")
    print("       FOOD COST")
    print("==============================")

    for item in result["breakdown"]:

        print(
            f"{item['ingredient']}: "
            f"{item['quantity']} × "
            f"₹{item['unit_price']} = "
            f"₹{item['cost']}"
        )

    print(
        "\nTotal estimated cost:",
        f"₹{result['total_cost']}"
    )