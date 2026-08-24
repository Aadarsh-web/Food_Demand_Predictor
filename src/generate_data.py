import pandas as pd
import numpy as np

# Generates 500 consecutive dates starting from January 1, 2026
dates = pd.date_range(start="2026-01-01", periods=500, freq="D")

# Randomly assign a meal type to each date
# Lunch is slightly more common than breakfast/dinner
meal_types = np.random.choice(
    ["Breakfast", "Lunch", "Dinner"],
    size=500,
    p=[0.3, 0.4, 0.3]
)

# Randomly assign one of five menu options to each meal
menus = np.random.choice(
    ["Chicken Rice", "Paneer Rice", "Dal Rice", "Rajma Rice", "Egg Curry Rice"],
    size=500
)

# Generate random expected attendance between 400 and 700 students
attendance = np.random.randint(400, 701, size=500)

# Generate environmental and calendar-related conditions for each meal event.
# Binary variables use 0 = No and 1 = Yes.
rain = np.random.choice([0, 1], size=500, p=[0.7, 0.3])
temperature = np.random.randint(20, 39, size=500)
holiday = np.random.choice([0, 1], size=500, p=[0.85, 0.15])
exam = np.random.choice([0, 1], size=500, p=[0.8, 0.2])
campus_event = np.random.choice([0, 1], size=500, p=[0.9, 0.1])
day_of_week = dates.dayofweek

# Generate realistic meal consumption based on attendance and other conditions.
# Attendance is the main factor, while weather, menu, exams, holidays, and events
# introduce smaller variations. Random noise prevents the relationship from being perfect.
menu_effect = {
    "Chicken Rice": 12,
    "Paneer Rice": 6,
    "Dal Rice": -2,
    "Rajma Rice": -5,
    "Egg Curry Rice": 10
}

meal_type_effect = {
    "Breakfast": -25,
    "Lunch": 10,
    "Dinner": 0
}

day_effect = {
    0: 0,    # Monday
    1: -5,   # Tuesday
    2: 3,    # Wednesday
    3: -3,   # Thursday
    4: 8,    # Friday
    5: 2,    # Saturday
    6: -12   # Sunday
}

consumption = (
    attendance
    + np.array([menu_effect[menu] for menu in menus])
    + np.array([meal_type_effect[meal] for meal in meal_types])
    + np.array([day_effect[day] for day in day_of_week])
    - rain * 15
    - exam * 10
    - holiday * 20
    + campus_event * 5
    + np.random.normal(0, 15, size=500)
)

#Prevents negative results from occuring
meals_consumed = np.clip(consumption, 0, attendance)


# Assume the kitchen's historical preparation is 10% above expected attendance.
prepared_quantity = np.ceil(attendance * 1.10)

# Calculate the previous day's surplus.
previous_surplus = np.roll(prepared_quantity - meals_consumed, 1)
previous_surplus[0] = np.mean(prepared_quantity - meals_consumed)

# Create historical features using the previous day's results.
# The first day has no previous record, so we fill it with the average values.
previous_consumption = np.roll(meals_consumed, 1)
previous_consumption[0] = np.mean(meals_consumed)


# Combine all generated variables into one structured dataset.
df = pd.DataFrame({
    "date": dates,
    "day_of_week": day_of_week,
    "attendance": attendance,
    "meal_type": meal_types,
    "menu": menus,
    "rain": rain,
    "temperature": temperature,
    "holiday": holiday,
    "exam": exam,
    "campus_event": campus_event,
    "previous_consumption": previous_consumption,
    "previous_surplus": previous_surplus,
    "meals_consumed": meals_consumed
})

#Removed the line to print sample data and table details

# Save the generated dataset as a CSV file so it can be used by our ML experiments.
df.to_csv("data/meal_data.csv", index=False)

# Confirm that the dataset was successfully saved.
print("\nDataset saved to data/meal_data.csv")