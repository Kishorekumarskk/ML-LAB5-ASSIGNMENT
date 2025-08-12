import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")

# Select all attributes except target
target_column = "restingBP"
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


reg_multi = LinearRegression().fit(X_train, y_train)

y_train_pred = reg_multi.predict(X_train)
y_test_pred = reg_multi.predict(X_test)


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4)
    }


train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

print("---- A3: Multiple Attribute Model ----")
print("Train Metrics:", train_metrics)
print("Test Metrics:", test_metrics)
