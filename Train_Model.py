import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------- LOAD DATA ----------------
df = pd.read_csv("student_data.csv")

# ---------------- FEATURES & TARGET ----------------
X = df.drop("final_score", axis=1)
y = df["final_score"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- MODEL ----------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test_scaled)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ---------------- SAVE ----------------
joblib.dump(model, "student_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully ✅")


