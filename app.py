from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
app = Flask(__name__)

MODEL_FILE   = "aqi_model.pkl"
CLEAN_FILE   = "karachi_clean_dataset.csv"
LAT          = 24.8607
LON          = 67.0011
OPENWEATHER_KEY = os.environ.get("OPENWEATHER_KEY", "")

FEATURES = [
    "pm25", "pm10", "no2", "co", "o3", "so2", "nh3",
    "hour", "day_of_week", "month",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_change"
]

AQI_LABELS = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}

# Load model once at startup
print("Loading model...")
model = joblib.load(MODEL_FILE)
print("Model loaded.")

df = pd.read_csv(CLEAN_FILE)
df.columns = df.columns.str.strip().str.lower()
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Dataset loaded: {len(df)} rows")

# =========================
# HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Karachi AQI Predictor API",
        "version": "1.0",
        "endpoints": ["/health", "/predict", "/forecast", "/current"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "Random Forest Classifier",
        "accuracy": "99.66%",
        "dataset_rows": len(df),
        "timestamp": datetime.utcnow().isoformat()
    })

# =========================
# PREDICT — manual input
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
    {
        "pm25": 25.0,
        "pm10": 60.0,
        "no2": 0.05,
        "co": 100.0,
        "o3": 90.0,
        "so2": 0.3,
        "nh3": 0.1,
        "hour": 14,
        "day_of_week": 2,
        "month": 5,
        "aqi_lag_1h": 3,
        "aqi_lag_3h": 3,
        "aqi_change": 0.0
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        row = pd.DataFrame([{f: float(data[f]) for f in FEATURES}])
        pred = int(model.predict(row)[0])
        pred = max(1, min(5, pred))

        proba = model.predict_proba(row)[0]
        confidence = float(round(max(proba) * 100, 2))

        return jsonify({
            "predicted_aqi": pred,
            "label": AQI_LABELS[pred],
            "confidence_pct": confidence,
            "hazardous": pred >= 4,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# CURRENT — from last dataset row
# =========================
@app.route("/current", methods=["GET"])
def current():
    try:
        last = df.iloc[-1]
        aqi  = int(last["aqi"])
        return jsonify({
            "aqi":         aqi,
            "label":       AQI_LABELS.get(aqi, "Unknown"),
            "pm25":        round(float(last["pm25"]), 2),
            "pm10":        round(float(last["pm10"]), 2),
            "no2":         round(float(last["no2"]), 4),
            "co":          round(float(last["co"]), 2),
            "o3":          round(float(last["o3"]), 2),
            "hazardous":   aqi >= 4,
            "last_updated": str(last["timestamp"]),
            "source":      "karachi_clean_dataset.csv"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# FORECAST — next 72 hours
# =========================
@app.route("/forecast", methods=["GET"])
def forecast():
    try:
        hours = int(request.args.get("hours", 72))
        hours = min(hours, 72)

        working = df.tail(10).copy().reset_index(drop=True)
        now     = datetime.utcnow()
        results = []

        for i in range(1, hours + 1):
            future   = now + timedelta(hours=i)
            last1    = working.iloc[-1]
            lag3_aqi = working["aqi"].iloc[-3] if len(working) >= 3 else working["aqi"].iloc[0]
            prev_aqi = working["aqi"].iloc[-2] if len(working) >= 2 else working["aqi"].iloc[0]

            row = pd.DataFrame([{
                "pm25":        float(last1["pm25"]),
                "pm10":        float(last1["pm10"]),
                "no2":         float(last1["no2"]),
                "co":          float(last1["co"]),
                "o3":          float(last1["o3"]),
                "so2":         float(last1["so2"]),
                "nh3":         float(last1["nh3"]),
                "hour":        future.hour,
                "day_of_week": future.weekday(),
                "month":       future.month,
                "aqi_lag_1h":  float(last1["aqi"]),
                "aqi_lag_3h":  float(lag3_aqi),
                "aqi_change":  float(last1["aqi"]) - float(prev_aqi),
            }])

            pred = int(model.predict(row)[0])
            pred = max(1, min(5, pred))

            results.append({
                "hour":          i,
                "datetime_utc":  future.isoformat(),
                "predicted_aqi": pred,
                "label":         AQI_LABELS[pred],
                "hazardous":     pred >= 4
            })

            new_row        = last1.copy()
            new_row["aqi"] = pred
            working        = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)

        hazardous_hours = sum(1 for r in results if r["hazardous"])

        return jsonify({
            "city":            "Karachi, Pakistan",
            "forecast_hours":  hours,
            "generated_at":    now.isoformat(),
            "hazardous_hours": hazardous_hours,
            "forecast":        results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("\n🌫️  Karachi AQI Predictor API")
    print("="*40)
    print("Endpoints:")
    print("  GET  /          → API info")
    print("  GET  /health    → Health check")
    print("  GET  /current   → Current AQI")
    print("  GET  /forecast  → 72hr forecast")
    print("  POST /predict   → Custom prediction")
    print("="*40)
    app.run(debug=True, host="0.0.0.0", port=5000)
