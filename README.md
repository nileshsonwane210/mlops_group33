# mlops_group33
# MLOps Assignment Summary: California Housing Predictor Pipeline

## Project Overview
This project implements an end-to-end MLOps pipeline for predicting median house values using the California Housing dataset. The pipeline integrates data preparation, model training and tracking, API serving, containerization, CI/CD automation, and logging/monitoring, demonstrating best practices for reproducible ML workflows.

Key components:
- **Data Preparation**: Used DVC for version control of datasets (e.g., train/test splits stored with hash `39278b72c2f5066520e78b558894aba7.dir`). Data is loaded, scaled with StandardScaler, and saved as CSV files.
- **Model Training and Tracking**: Trained baseline models (Linear Regression and Decision Tree Regressor) using scikit-learn, tracked experiments in MLflow with metrics (RMSE, R2), parameters, and artifacts. Models and scaler are saved via joblib for deployment.
- **API Serving**: Built a FastAPI app (`app.py`) with a `/predict` endpoint for real-time predictions. Input validation uses Pydantic, and the model/scaler are loaded from files.
- **Containerization**: Dockerized the app in a `Dockerfile` based on Python 3.9-slim, copying dependencies, source code, model, and scaler. Image is tagged and pushed to Docker Hub.
- **CI/CD Automation**: GitHub Actions workflow (`ci-cd.yml`) triggers on pushes to `rs_05021`, running linting (flake8), Docker build/push (using secrets), and a local deployment simulation.
- **Logging and Monitoring**: Added request/response logging to `app.log` via Python's logging module, stored prediction details in SQLite (`logs.db`), and an optional `/metrics` endpoint with Prometheus for request counts.

The pipeline ensures reproducibility, automation, and monitoring, from data to deployment.

## Challenges and Solutions
Several hurdles arose during implementation, mainly around setup, debugging, and compatibility:
- **Repo Restructuring**: Moved files from a subfolder to root for Actions detection; resolved virtual env path issues by recreating `mlops_env` and reinstalling dependencies.
- **Linting and Testing Errors**: Flake8 failed on unused imports, whitespace, and long lines—fixed iteratively by editing files like `app.py`, `data_prep.py`, and `train.py`. Pytest errored with no tests—added a dummy `test_dummy.py` to pass.
- **Docker Build Failures**: Missing files (`scaler.pkl`, `model/`) caused "not found" errors—generated via training script, force-added to Git despite `.gitignore`.
- **App Runtime Issues**: SQLite thread safety errors in FastAPI—resolved by creating connections per request. Sklearn warnings on feature names—fixed by converting DataFrame to NumPy array before prediction.
- **Other**: Pip conflicts (e.g., built-in `logging`), Git reverts, and env activations—addressed with targeted commands and fresh recreations.

These challenges highlighted the importance of clean code, thread safety in async apps, and proper Git management.

## Results and Learnings
- **Model Performance**: Decision Tree (max_depth=5) achieved RMSE ~0.75 and R2 ~0.60 on test data; Linear Regression had RMSE ~0.72 and R2 ~0.61 (tracked in MLflow UI).
- **Pipeline Outcomes**: Successful CI/CD runs build and push the image to Docker Hub, enabling quick deployments. Logging captures real-time data for auditing (viewable via `sqlite3 logs.db` or file inspection).
- **Key Learnings**: MLOps requires iterative debugging (e.g., linting loops taught code hygiene); tools like MLflow and DVC enhance reproducibility; containerization and automation streamline scaling. Overall, this built skills in integrating ML with DevOps practices.