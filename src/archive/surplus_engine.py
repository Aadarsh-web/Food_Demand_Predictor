from datetime import datetime


# =========================================================
# SAFETY CONFIGURATION
# =========================================================
#
# These are prototype thresholds based on FSSAI guidance.
# They MUST be reviewed against the applicable current
# regulations before real-world deployment.
#
# Hot food:
#   Target holding temperature: 65°C or above
#
# Cold food:
#   Target holding temperature: 5°C or below
#
# We deliberately make the system conservative.
# =========================================================

HOT_MIN_TEMP = 65
COLD_MAX_TEMP = 5

MAX_SURPLUS_AGE_HOURS = 2


# =========================================================
# RECORD SURPLUS FOOD
# =========================================================

def record_surplus(
    menu,
    prepared_quantity,
    consumed_quantity,
    storage_temperature,
    storage_type,
    prepared_time
):
    """
    Record the surplus remaining after a meal.

    storage_type:
        "hot"  -> food is being held hot
        "cold" -> food is being refrigerated

    Returns a structured surplus record.
    """

    surplus_quantity = max(
        0,
        prepared_quantity - consumed_quantity
    )

    return {
        "menu": menu,
        "prepared_quantity": prepared_quantity,
        "consumed_quantity": consumed_quantity,
        "surplus_quantity": surplus_quantity,
        "storage_temperature": storage_temperature,
        "storage_type": storage_type,
        "prepared_time": prepared_time
    }


# =========================================================
# CHECK FOOD SAFETY
# =========================================================

def check_food_safety(surplus):
    """
    Perform a conservative prototype safety check.

    This does NOT certify food as safe.
    It only determines whether the record passes
    the application's configured checks.
    """

    quantity = surplus["surplus_quantity"]
    temperature = surplus["storage_temperature"]
    storage_type = surplus["storage_type"]

    if quantity <= 0:
        return {
            "status": "NO_SURPLUS",
            "reason": "There is no surplus food."
        }

    # Check temperature requirements.
    if storage_type == "hot":

        if temperature < HOT_MIN_TEMP:
            return {
                "status": "REJECT",
                "reason": (
                    f"Hot food temperature is below "
                    f"{HOT_MIN_TEMP}°C."
                )
            }

    elif storage_type == "cold":

        if temperature > COLD_MAX_TEMP:
            return {
                "status": "REJECT",
                "reason": (
                    f"Cold food temperature is above "
                    f"{COLD_MAX_TEMP}°C."
                )
            }

    else:

        return {
            "status": "REVIEW",
            "reason": "Unknown storage type."
        }

    # If all configured checks pass.
    return {
        "status": "ELIGIBLE_FOR_REVIEW",
        "reason": "Temperature requirements passed."
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    surplus = record_surplus(
        menu="Chicken Rice",
        prepared_quantity=587,
        consumed_quantity=555,
        storage_temperature=68,
        storage_type="hot",
        prepared_time=datetime.now()
    )

    safety_result = check_food_safety(surplus)

    print("\n==============================")
    print("       SURPLUS CHECK")
    print("==============================")

    print(
        "Menu:",
        surplus["menu"]
    )

    print(
        "Surplus:",
        surplus["surplus_quantity"],
        "servings"
    )

    print(
        "Temperature:",
        surplus["storage_temperature"],
        "°C"
    )

    print(
        "Safety status:",
        safety_result["status"]
    )

    print(
        "Reason:",
        safety_result["reason"]
    )