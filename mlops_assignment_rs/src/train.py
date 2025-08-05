import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path
from mlflow.models.signature import infer_signature
import joblib  # For serializing the scaler

# Set paths (adjust if your data/ is elsewhere)
data_dir = Path(__file__).parent.parent / 'data'

# Load preprocessed data
X_train = pd.read_csv(data_dir / 'train_data.csv')
y_train = pd.read_csv(data_dir / 'train_labels.csv')['MedHouseVal']
X_test = pd.read_csv(data_dir / 'test_data.csv')
y_test = pd.read_csv(data_dir / 'test_labels.csv')['MedHouseVal']

# Ensure scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set a named experiment for better organization in UI
mlflow.set_experiment("California_Housing_Regression")

# Parent run for the experiment
with mlflow.start_run() as parent_run:
    print(f"Parent run ID: {parent_run.info.run_id}")
    
    # Log scaler as artifact (serialize with joblib)
    scaler_path = "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path)  # Log the pickled scaler file
    
    # Log dataset metadata (integrate with DVC if you have a version hash)
    mlflow.log_param("dataset_source", "sklearn.fetch_california_housing")
    mlflow.log_param("dataset_version_DVC_Hash", "39278b72c2f5066520e78b558894aba7.dir")  
    mlflow.set_tag("author", "Group 33")  
    mlflow.set_tag("description", "Baseline regression models for housing prices")
    
    # Sample input for signature inference (use a row from X_test)
    input_example = X_test.iloc[0:1]  # Single row as DataFrame
    
    # Nested run for Model 1: Linear Regression
    with mlflow.start_run(nested=True):
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        lr_r2 = r2_score(y_test, lr_pred)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("random_state", "None")  # Document for consistency
        mlflow.log_metric("rmse", lr_rmse)
        mlflow.log_metric("r2", lr_r2)
        signature = infer_signature(X_test, lr_pred)  # Infer schema
        mlflow.sklearn.log_model(lr, "linear_model", signature=signature, input_example=input_example)
        mlflow.set_tag("run_type", "baseline_linear")
    
    # Nested run for Model 2: Decision Tree Regressor
    with mlflow.start_run(nested=True):
        dt = DecisionTreeRegressor(max_depth=5, random_state=42)
        dt.fit(X_train_scaled, y_train)
        dt_pred = dt.predict(X_test_scaled)
        dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
        dt_r2 = r2_score(y_test, dt_pred)
        mlflow.log_param("model_type", "DecisionTreeRegressor")
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("rmse", dt_rmse)
        mlflow.log_metric("r2", dt_r2)
        signature = infer_signature(X_test, dt_pred)  # Infer schema
        mlflow.sklearn.log_model(dt, "dt_model", signature=signature, input_example=input_example)
        mlflow.set_tag("run_type", "baseline_tree")

print("Training complete. View runs with 'mlflow ui'.")
