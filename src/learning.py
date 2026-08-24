import pandas as pd
from model import train_model


DATA_PATH = "data/meal_data.csv"


def record_actual_result(
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
    predicted_consumption,
    actual_consumed
):
    """
    Add today's actual result to the historical dataset.

    The new record allows the model to learn from
    what actually happened.
    """

    # Load existing historical data.
    df = pd.read_csv(DATA_PATH)

    # Calculate how wrong today's prediction was.
    prediction_error = (
        actual_consumed - predicted_consumption
    )

    # Create the new historical record.
    new_record = {
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
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
        "previous_surplus": previous_surplus,
        "meals_consumed": actual_consumed
    }

    # Add the new observation to the dataset.
    df = pd.concat(
        [
            df,
            pd.DataFrame([new_record])
        ],
        ignore_index=True
    )

    # Save the updated dataset.
    df.to_csv(DATA_PATH, index=False)

    # Retrain the model using the newly updated dataset.
    train_model()

    return {
        "prediction": predicted_consumption,
        "actual": actual_consumed,
        "error": round(prediction_error, 2),
        "dataset_size": len(df)
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    result = record_actual_result(
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
        day_of_week=2,
        predicted_consumption=570,
        actual_consumed=555
    )

    print("\n==============================")
    print("      LEARNING RECORD")
    print("==============================")

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "Actual:",
        result["actual"]
    )

    print(
        "Prediction error:",
        result["error"]
    )

    print(
        "Dataset size:",
        result["dataset_size"]
    )