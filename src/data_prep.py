from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set up paths relative to the script's location
script_dir = Path(__file__).parent  # src/
data_dir = script_dir.parent / 'data'  # Goes up to root, then into data/

# Create data directory if it doesn't exist
data_dir.mkdir(parents=True, exist_ok=True)

# Load and preprocess the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
df = df.dropna()  # Handle missing values (e.g., total_bedrooms)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save to CSV in data/
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(data_dir / 'train_data.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(data_dir / 'test_data.csv', index=False)
pd.Series(y_train, name='MedHouseVal').to_csv(data_dir / 'train_labels.csv', index=False)
pd.Series(y_test, name='MedHouseVal').to_csv(data_dir / 'test_labels.csv', index=False)

print(f"Data saved successfully to {data_dir}")
