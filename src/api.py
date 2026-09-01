from fastapi import FastAPI
from pydantic import BaseModel

from annadata_engine import run_annadata


app = FastAPI(
    title="AnnaData AI Engine",
    version="1.0.0"
)


class MealRequest(BaseModel):
    attendance: int
    meal_type: str
    menu: str
    rain: int
    temperature: float
    holiday: int
    exam: int
    campus_event: int
    previous_consumption: float
    previous_surplus: float
    day_of_week: int
    month: int
    season: str

    actual_consumed: float = 0
    storage_temperature: float = 0
    storage_type: str = "hot"
    storage_duration_hours: float = 0

    hostel_latitude: float
    hostel_longitude: float
    location: str


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AnnaData AI Engine"
    }


@app.post("/predict")
def predict(meal: MealRequest):

    result = run_annadata(
        **meal.model_dump()
    )

    return result