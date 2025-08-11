from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_and_save_data(filepath="data/raw/housing.csv"):
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    load_and_save_data()
