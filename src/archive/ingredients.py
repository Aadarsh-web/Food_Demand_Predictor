import pandas as pd


def calculate_ingredients(menu, servings):
    """
    Calculate the ingredients required for a given number of servings.
    Recipe quantities are stored per 100 servings.
    """

    recipes = pd.read_csv("data/recipes.csv")

    # Get only the ingredients belonging to the selected menu.
    menu_recipe = recipes[recipes["menu"] == menu]

    if menu_recipe.empty:
        raise ValueError(f"No recipe found for: {menu}")

    results = []

    for _, row in menu_recipe.iterrows():

        # Scale the recipe from 100 servings to the required servings.
        quantity = row["quantity_per_100"] * servings / 100

        results.append({
            "ingredient": row["ingredient"],
            "quantity": round(quantity, 2)
        })

    return pd.DataFrame(results)


# Test the calculator.
result = calculate_ingredients(
    menu="Chicken Rice",
    servings=587
)

print("\n--- INGREDIENT REQUIREMENTS ---")
print(result.to_string(index=False))