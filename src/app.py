from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import sqlite3
from datetime import datetime
from prometheus_client import Counter, generate_latest  # For optional metrics

app = FastAPI()

# Setup logging to file
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Optional: Prometheus metrics
request_counter = Counter('prediction_requests_total', 'Total prediction requests')


# Middleware to log requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response: Response = await call_next(request)

    # Log to file
    log_data = {
        "timestamp": start_time.isoformat(),
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host,
        "status": response.status_code
    }
    logging.info(f"Request: {log_data}")

    # Increment metrics (optional)
    request_counter.inc()

    return response

# Your existing model loading (unchanged)
model = joblib.load("model/model.pkl")
scaler = joblib.load("scaler.pkl")


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
    data = pd.DataFrame([input.dict()])

    # Scale while keeping it as DataFrame with original columns (for reference)
    scaled_data = pd.DataFrame(
        scaler.transform(data),
        columns=data.columns  # Reattach the expected column names
    )

    # Convert to NumPy array without names to match model's training (fixes warning)
    scaled_data_no_names = scaled_data.to_numpy()

    # Make prediction (now matches the model's expected schema)
    prediction = model.predict(scaled_data_no_names)[0]  # Use [0] for single prediction

    # Log to SQLite (create connection per request for thread safety)
    conn = sqlite3.connect('logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (timestamp TEXT, input_data TEXT, output REAL)''')
    timestamp = datetime.now().isoformat()
    input_str = str(input.dict())  # Convert input to string for storage
    c.execute("INSERT INTO predictions (timestamp, input_data, output) VALUES (?, ?, ?)",
              (timestamp, input_str, prediction))
    conn.commit()
    conn.close()

    # Log output to file
    logging.info(f"Prediction output: {prediction} for input: {input_str}")

    return {"prediction": prediction}


# Optional: Expose /metrics endpoint for monitoring
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
