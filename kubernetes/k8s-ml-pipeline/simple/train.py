import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Dynamically determine the current directory
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, "dataset.csv")

print(f"Current directory: {current_dir}")
print(f"Dataset path: {dataset_path}")

# Load the dataset
try:
    data = pd.read_csv(dataset_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Features and target variable
X = data[["square_footage", "bedrooms", "bathrooms"]]
y = data["price"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
model_path = os.path.join(current_dir, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model trained and saved as {model_path}")
