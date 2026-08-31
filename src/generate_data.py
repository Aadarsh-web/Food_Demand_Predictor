import pandas as pd
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

np.random.seed(42)

START_DATE = "2010-01-01"
END_DATE = "2026-08-31"


# =========================================================
# 31 REAL FOOD ITEMS
# =========================================================

foods = [
    "Chole",
    "Bhature",
    "Idli",
    "Bada",
    "Ghuguni",
    "Masala Upma",
    "Poha",
    "Dry Ghuguni",
    "Onion Masala Dosa",
    "Sambar",
    "Dahi Vada",
    "Aloo Dum",
    "Vada Pav",
    "Aloo Chop",
    "Aloo Samosa",
    "Papdi Chaat",
    "Fried Chilli",
    "Boiled Egg",
    "Tea",
    "Coffee",
    "Dahi",
    "Milkshake",
    "Banana",
    "Onion",
    "Coconut Chutney",
    "Groundnut Chutney",
    "Pudina Chutney",
    "Bread",
    "Pav",
    "Cream Bun",
    "Sev"
]


# =========================================================
# FOOD POPULARITY
# =========================================================

food_popularity = {

    "Chole": 1.05,
    "Bhature": 1.08,
    "Idli": 1.02,
    "Bada": 0.95,
    "Ghuguni": 0.98,
    "Masala Upma": 0.88,
    "Poha": 1.00,
    "Dry Ghuguni": 0.90,
    "Onion Masala Dosa": 1.08,
    "Sambar": 1.00,
    "Dahi Vada": 0.92,
    "Aloo Dum": 0.98,
    "Vada Pav": 1.05,
    "Aloo Chop": 0.94,
    "Aloo Samosa": 1.02,
    "Papdi Chaat": 0.90,
    "Fried Chilli": 0.65,
    "Boiled Egg": 0.98,
    "Tea": 1.12,
    "Coffee": 0.90,
    "Dahi": 0.92,
    "Milkshake": 0.95,
    "Banana": 0.96,
    "Onion": 0.55,
    "Coconut Chutney": 0.80,
    "Groundnut Chutney": 0.82,
    "Pudina Chutney": 0.80,
    "Bread": 0.90,
    "Pav": 0.94,
    "Cream Bun": 0.82,
    "Sev": 0.72
}


# =========================================================
# REALISTIC MULTI-FOOD MEAL PATTERNS
# =========================================================

breakfast_patterns = [
    ["Idli", "Sambar", "Coconut Chutney"],
    ["Idli", "Sambar", "Groundnut Chutney"],
    ["Bada", "Sambar", "Coconut Chutney"],
    ["Poha", "Banana", "Tea"],
    ["Masala Upma", "Banana", "Tea"],
    ["Onion Masala Dosa", "Sambar", "Coconut Chutney"],
    ["Ghuguni", "Bread", "Tea"],
    ["Boiled Egg", "Bread", "Tea"],
    ["Boiled Egg", "Bread", "Coffee"],
    ["Vada Pav", "Tea", "Banana"],
    ["Idli", "Sambar", "Pudina Chutney"],
    ["Poha", "Tea", "Banana"],
    ["Masala Upma", "Tea", "Banana"],
    ["Bada", "Sambar", "Pudina Chutney"],
    ["Cream Bun", "Tea", "Banana"],
    ["Cream Bun", "Coffee", "Banana"],
    ["Pav", "Boiled Egg", "Tea"],
    ["Pav", "Tea", "Sev"],
    ["Milkshake", "Bread", "Banana"]
]

lunch_patterns = [

    ["Chole", "Bhature", "Onion"],
    ["Chole", "Bhature", "Dahi", "Onion"],
    ["Ghuguni", "Aloo Dum", "Onion", "Dahi"],
    ["Aloo Dum", "Dahi", "Onion", "Sev"],
    ["Onion Masala Dosa", "Sambar", "Coconut Chutney", "Dahi"],
    ["Vada Pav", "Sev", "Onion", "Dahi"],
    ["Aloo Chop", "Dahi", "Onion", "Sev"],
    ["Aloo Samosa", "Dahi", "Onion", "Pudina Chutney"],
    ["Papdi Chaat", "Dahi", "Sev", "Pudina Chutney"],
    ["Dry Ghuguni", "Aloo Dum", "Onion", "Dahi"],
    ["Chole", "Bhature", "Dahi", "Pudina Chutney"],
    ["Ghuguni", "Aloo Chop", "Onion", "Dahi"],
    ["Vada Pav", "Aloo Chop", "Sev", "Onion"],
    ["Chole", "Bhature", "Dahi", "Fried Chilli", "Onion"],
    ["Aloo Dum", "Dahi", "Fried Chilli", "Onion"],
    ["Aloo Samosa", "Dahi", "Pudina Chutney", "Fried Chilli"],
    ["Papdi Chaat", "Dahi", "Sev", "Fried Chilli"],
    ["Vada Pav", "Pav", "Sev", "Onion"]
]


dinner_patterns = [

    ["Chole", "Bhature", "Onion"],
    ["Chole", "Bhature", "Dahi", "Onion"],
    ["Ghuguni", "Aloo Dum", "Onion", "Dahi"],
    ["Aloo Dum", "Dahi", "Onion", "Sev"],
    ["Onion Masala Dosa", "Sambar", "Coconut Chutney", "Dahi"],
    ["Vada Pav", "Sev", "Onion", "Dahi"],
    ["Aloo Chop", "Dahi", "Onion", "Sev"],
    ["Aloo Samosa", "Dahi", "Pudina Chutney", "Onion"],
    ["Papdi Chaat", "Dahi", "Sev", "Pudina Chutney"],
    ["Dry Ghuguni", "Aloo Dum", "Onion", "Dahi"],
    ["Ghuguni", "Aloo Chop", "Onion", "Dahi"],
    ["Chole", "Bhature", "Dahi", "Pudina Chutney"],
    ["Chole", "Bhature", "Dahi", "Fried Chilli", "Onion"],
    ["Aloo Dum", "Dahi", "Fried Chilli", "Onion"],
    ["Aloo Samosa", "Dahi", "Pudina Chutney", "Fried Chilli"],
    ["Papdi Chaat", "Dahi", "Sev", "Fried Chilli"],
    ["Vada Pav", "Pav", "Sev", "Onion"]
]


# =========================================================
# DATE RANGE
# =========================================================

dates = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq="D"
)


records = []


# Previous values.
previous_consumption = 500
previous_surplus = 20


# =========================================================
# DAILY SIMULATION
# =========================================================

for date in dates:

    day_of_week = date.dayofweek
    month = date.month
    year = date.year


    # -----------------------------------------------------
    # SEASON
    # -----------------------------------------------------

    if month in [12, 1, 2]:
        season = "Winter"

    elif month in [3, 4, 5]:
        season = "Summer"

    elif month in [6, 7, 8, 9]:
        season = "Monsoon"

    else:
        season = "Post-Monsoon"


    # -----------------------------------------------------
    # TEMPERATURE
    # -----------------------------------------------------

    seasonal_temperature = {

        "Winter": 20,
        "Summer": 32,
        "Monsoon": 28,
        "Post-Monsoon": 27
    }

    temperature = np.random.normal(
        seasonal_temperature[season],
        3.5
    )

    temperature = np.clip(
        temperature,
        12,
        42
    )


    # -----------------------------------------------------
    # RAIN
    # -----------------------------------------------------

    if season == "Monsoon":

        rain = np.random.choice(
            [0, 1],
            p=[0.40, 0.60]
        )

    elif season == "Post-Monsoon":

        rain = np.random.choice(
            [0, 1],
            p=[0.75, 0.25]
        )

    else:

        rain = np.random.choice(
            [0, 1],
            p=[0.90, 0.10]
        )


    # -----------------------------------------------------
    # HOLIDAY / EXAM / EVENT
    # -----------------------------------------------------

    holiday = np.random.choice(
        [0, 1],
        p=[0.91, 0.09]
    )

    exam = np.random.choice(
        [0, 1],
        p=[0.82, 0.18]
    )

    campus_event = np.random.choice(
        [0, 1],
        p=[0.90, 0.10]
    )


    # -----------------------------------------------------
    # BASE HOSTEL ATTENDANCE
    # -----------------------------------------------------

    year_progress = (
        year - 2010
    ) / (2026 - 2010)

    base_students = (
        500
        + year_progress * 180
    )

    daily_attendance = np.random.normal(
        base_students,
        30
    )


    # Weekend effect.
    if day_of_week == 5:
        daily_attendance -= 30

    elif day_of_week == 6:
        daily_attendance -= 80


    # Holiday effect.
    if holiday:

        daily_attendance *= np.random.uniform(
            0.35,
            0.65
        )


    # Exam effect.
    if exam:

        daily_attendance *= np.random.uniform(
            0.88,
            0.96
        )


    # Campus event.
    if campus_event:

        daily_attendance *= np.random.uniform(
            1.03,
            1.15
        )


    # Rain.
    if rain:

        daily_attendance *= np.random.uniform(
            0.94,
            0.98
        )


    daily_attendance = int(
        np.clip(
            round(daily_attendance),
            100,
            800
        )
    )


    # =====================================================
    # THREE MEALS EVERY DAY
    # =====================================================

    meal_patterns = {

        "Breakfast": breakfast_patterns,
        "Lunch": lunch_patterns,
        "Dinner": dinner_patterns
    }


    for meal_type in [
        "Breakfast",
        "Lunch",
        "Dinner"
    ]:


        # -------------------------------------------------
        # SELECT ONE MULTI-FOOD MENU
        # -------------------------------------------------

        if meal_type == "Breakfast":

            pattern_index = np.random.randint(
                len(breakfast_patterns)
            )

            selected_foods = breakfast_patterns[
                pattern_index
            ]


        elif meal_type == "Lunch":

            pattern_index = np.random.randint(
                len(lunch_patterns)
            )

            selected_foods = lunch_patterns[
                pattern_index
            ]


        else:

            pattern_index = np.random.randint(
                len(dinner_patterns)
            )

            selected_foods = dinner_patterns[
                pattern_index
            ]


        # -------------------------------------------------
        # MEAL ATTENDANCE
        # -------------------------------------------------

        if meal_type == "Breakfast":

            meal_attendance = (
                daily_attendance
                * np.random.uniform(
                    0.65,
                    0.82
                )
            )

        elif meal_type == "Lunch":

            meal_attendance = (
                daily_attendance
                * np.random.uniform(
                    0.88,
                    0.98
                )
            )

        else:

            meal_attendance = (
                daily_attendance
                * np.random.uniform(
                    0.82,
                    0.95
                )
            )


        meal_attendance = int(
            np.clip(
                round(meal_attendance),
                50,
                daily_attendance
            )
        )


        # -------------------------------------------------
        # GENERATE EACH FOOD
        # -------------------------------------------------

        for food in selected_foods:

            popularity = food_popularity[food]


            # Base consumption rate.
            if meal_type == "Breakfast":

                base_rate = 0.80

            elif meal_type == "Lunch":

                base_rate = 0.94

            else:

                base_rate = 0.89


            consumption_rate = (
                base_rate
                * popularity
            )


            # -------------------------------------------------
            # SIDE DISHES
            # -------------------------------------------------

            if food in [
                "Onion",
                "Coconut Chutney",
                "Groundnut Chutney",
                "Pudina Chutney",
                "Sev"
            ]:

                consumption_rate *= np.random.uniform(
                    0.55,
                    0.85
                )


            # -------------------------------------------------
            # WEATHER
            # -------------------------------------------------

            if rain:

                consumption_rate *= np.random.uniform(
                    0.96,
                    0.99
                )


            # Cold weather increases tea demand.
            if food == "Tea" and temperature < 18:

                consumption_rate *= 1.10


            # Hot weather increases milkshake demand.
            if food == "Milkshake" and temperature > 34:

                consumption_rate *= 1.08


            # -------------------------------------------------
            # WEEKDAY
            # -------------------------------------------------

            if day_of_week == 6:

                consumption_rate *= 0.96

            elif day_of_week == 4:

                consumption_rate *= 1.02


            # -------------------------------------------------
            # HOLIDAY
            # -------------------------------------------------

            if holiday:

                consumption_rate *= np.random.uniform(
                    0.90,
                    0.98
                )


            # -------------------------------------------------
            # EVENT
            # -------------------------------------------------

            if campus_event:

                consumption_rate *= np.random.uniform(
                    1.01,
                    1.06
                )


            # -------------------------------------------------
            # PREVIOUS DEMAND
            # -------------------------------------------------

            historical_rate = (
                previous_consumption
                / max(
                    meal_attendance,
                    1
                )
            )

            historical_rate = np.clip(
                historical_rate,
                0.65,
                1.00
            )


            consumption_rate = (
                consumption_rate * 0.85
                + historical_rate * 0.15
            )


            # -------------------------------------------------
            # RANDOM NOISE
            # -------------------------------------------------

            consumption_rate += np.random.normal(
                0,
                0.035
            )


            # -------------------------------------------------
            # RARE UNUSUAL EVENT
            # -------------------------------------------------

            if np.random.random() < 0.015:

                consumption_rate += np.random.uniform(
                    -0.15,
                    0.15
                )


            # -------------------------------------------------
            # LIMIT
            # -------------------------------------------------

            consumption_rate = np.clip(
                consumption_rate,
                0.30,
                1.05
            )


            # -------------------------------------------------
            # FINAL CONSUMPTION
            # -------------------------------------------------

            consumption = (
                meal_attendance
                * consumption_rate
            )


            consumption += np.random.normal(
                0,
                6
            )


            meals_consumed = int(
                np.clip(
                    round(consumption),
                    0,
                    meal_attendance
                )
            )


            # -------------------------------------------------
            # PREPARATION
            # -------------------------------------------------

            prepared_quantity = int(
                np.ceil(
                    meal_attendance * 1.08
                )
            )


            current_surplus = max(
                0,
                prepared_quantity
                - meals_consumed
            )


            # -------------------------------------------------
            # SAVE ROW
            # -------------------------------------------------

            records.append({

                "date": date,

                "day_of_week": day_of_week,

                "month": month,

                "season": season,

                "attendance": meal_attendance,

                "meal_type": meal_type,

                "menu": food,

                "rain": rain,

                "temperature": round(
                    temperature,
                    1
                ),

                "holiday": holiday,

                "exam": exam,

                "campus_event": campus_event,

                "previous_consumption": int(
                    previous_consumption
                ),

                "previous_surplus": int(
                    previous_surplus
                ),

                "meals_consumed": meals_consumed
            })


            # Update historical values.
            previous_consumption = meals_consumed
            previous_surplus = current_surplus


# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(records)


df = df.sort_values(
    [
        "date",
        "meal_type"
    ]
).reset_index(
    drop=True
)


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    "data/meal_data.csv",
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print(
    f"\nGenerated {len(df)} food-meal records."
)

print(
    f"Date range: "
    f"{df['date'].min().date()} "
    f"to "
    f"{df['date'].max().date()}"
)

print(
    "\nRecords by meal type:"
)

print(
    df["meal_type"].value_counts()
)

print(
    "\nFood distribution:"
)

print(
    df["menu"].value_counts()
)

print(
    "\nAverage foods per day:"
)

print(
    round(
        df.groupby("date")["menu"].count().mean(),
        2
    )
)

print(
    "\nDataset saved to data/meal_data.csv"
)