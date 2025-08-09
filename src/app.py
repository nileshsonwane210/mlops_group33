from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI()


# Load the registered MLflow model (update with your exact model URI if needed)
model_uri = "models:/california_housing_best_dt_model/1"  # Assumes version 1; check your MLflow UI
# model = mlflow.pyfunc.load_model(model_uri)
model = joblib.load("model/model.pkl")


# Load the scaler from local file (downloaded from MLflow)
scaler = joblib.load("scaler.pkl")  # Assumes it's in the repo root; adjust path if needed


# Define input schema with Pydantic (for validation - bonus points!)


class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.post("/predict")
def predict(input: HousingInput):
    # Convert input to DataFrame (preserves column names)
    data = pd.DataFrame([input.dict()])

    # Scale while keeping it as DataFrame with original columns
    scaled_data = pd.DataFrame(
        scaler.transform(data),
        columns=data.columns  # Reattach the expected column names
    )

    # Make prediction (now matches the model's expected schema)
    prediction = model.predict(scaled_data)

    return {"prediction": prediction[0]}  # Returns median house value
