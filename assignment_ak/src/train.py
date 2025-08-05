import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.preprocess import preprocess

def train_and_log_models():
    """
    Train models and log results to MLflow.
    """
    X_train, X_test, y_train, y_test = preprocess()

    # Enable MLflow auto-logging
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("California-Housing-ML")

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
    }

    for model_name, model in models.items():
        try:
            with mlflow.start_run(run_name=model_name):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                mse = mean_squared_error(y_test, predictions)
                mlflow.log_param("model_type", model_name)
                mlflow.log_metric("mse", mse)
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    input_example=X_test[:5],
                    signature=mlflow.models.infer_signature(X_test, predictions)
                )

                print(f"{model_name} MSE: {mse}")
        except Exception as e:
            print(f"Error training {model_name}: {e}")

if __name__ == "__main__":
    train_and_log_models()