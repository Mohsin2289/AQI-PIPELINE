import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# =========================
# CONFIG
# =========================
CLEAN_FILE = "karachi_clean_dataset.csv"
MODEL_FILE = "aqi_model.pkl"

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_csv(CLEAN_FILE)
df.columns = df.columns.str.strip().str.lower()

print(f"Total rows: {len(df)}")
print(f"AQI distribution:\n{df['aqi'].value_counts().sort_index()}")

# =========================
# FEATURES & TARGET
# =========================
FEATURES = [
    "pm25", "pm10", "no2", "co", "o3", "so2", "nh3",
    "hour", "day_of_week", "month",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_change"
]

TARGET = "aqi"

# Drop rows where features are missing
df = df.dropna(subset=FEATURES + [TARGET])
print(f"\nRows after dropping NaN: {len(df)}")

X = df[FEATURES]
y = df[TARGET]

# =========================
# TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

# =========================
# TRAIN MODEL
# =========================
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# FEATURE IMPORTANCE
# =========================
print("\nFeature Importances:")
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)
for feat, score in importances.items():
    print(f"  {feat:20s}: {score:.4f}")
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"\nRMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
# =========================
# SAVE MODEL
# =========================
joblib.dump(model, MODEL_FILE)
print(f"\nModel saved: {MODEL_FILE}")