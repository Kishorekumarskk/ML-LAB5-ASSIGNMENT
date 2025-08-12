# A2 - Compare metrics between train and test sets

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------
# Function to train and predict using Linear Regression
# ------------------------------
def train_linear_regression(X_train, y_train, X_test):
    """Train a Linear Regression model and predict on both train and test data."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return y_train_pred, y_test_pred, model

# ------------------------------
# Function to evaluate predictions
# ------------------------------
def evaluate_model(y_true, y_pred):
    """Evaluate the model using MSE, RMSE, MAPE, and R2 score, rounded for readability."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "R2": round(float(r2), 4)
    }

# ------------------------------
# Main Program - A2
# ------------------------------
if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
    
    # Single attribute: Age; Target: RestingBP
    X = dataset[["age"]].values
    y = dataset["restingBP"].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model and get predictions
    y_train_pred, y_test_pred, _ = train_linear_regression(X_train, y_train, X_test)
    
    # Evaluate both sets
    train_metrics = evaluate_model(y_train, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    # Compare metrics
    print("---- A2: Metrics Comparison ----")
    print("Train Metrics:", train_metrics)
    print("Test Metrics :", test_metrics)
    
    # Optional: direct difference comparison
    print("\nMetric Differences (Train - Test):")
    for metric in train_metrics.keys():
        diff = round(train_metrics[metric] - test_metrics[metric], 4)
        print(f"{metric}: {diff}")
