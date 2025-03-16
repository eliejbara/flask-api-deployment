# demand_prediction.py
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load("demand_model.pkl")

# The same dummy month columns and the same FEATURES list
DUMMY_MONTH_COLS = [f"month_{m}" for m in range(2, 13)]
FEATURES = [
    "year",
    "day_of_week",
    "is_weekend",
    "is_holiday_season",
    "avg_lead_time",
    "sum_previous_bookings",
    "avg_adr",
    "total_children"
] + DUMMY_MONTH_COLS

@app.route('/predict_demand', methods=['GET'])
def predict_demand():
    """
    Example usage:
    curl "http://127.0.0.1:5000/predict_demand?year=2025&month=4&day_of_week=2&is_weekend=0&is_holiday_season=0&avg_lead_time=30&sum_previous_bookings=10&avg_adr=100&total_children=2"
    """
    try:
        # 1) Parse input
        year = int(request.args.get("year", 2025))
        month = int(request.args.get("month", 7))  # numeric 1..12
        day_of_week = int(request.args.get("day_of_week", 4))
        is_weekend = int(request.args.get("is_weekend", 0))
        is_holiday_season = int(request.args.get("is_holiday_season", 0))
        avg_lead_time = float(request.args.get("avg_lead_time", 30))
        sum_previous_bookings = float(request.args.get("sum_previous_bookings", 5))
        avg_adr = float(request.args.get("avg_adr", 100))
        total_children = float(request.args.get("total_children", 2))

        # 2) Build row_dict
        row_dict = {
            "year": year,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_holiday_season": is_holiday_season,
            "avg_lead_time": avg_lead_time,
            "sum_previous_bookings": sum_previous_bookings,
            "avg_adr": avg_adr,
            "total_children": total_children
        }

        # 3) Initialize dummy month cols
        for m_col in DUMMY_MONTH_COLS:
            row_dict[m_col] = 0

        # If month=1 => baseline (month_1 is dropped)
        # If month >=2 => set that dummy col to 1
        month_col = f"month_{month}"
        if month_col in row_dict:
            row_dict[month_col] = 1

        # 4) Create DataFrame in EXACT same column order
        X_input = pd.DataFrame([row_dict])
        X_input = X_input[FEATURES]  # reorder columns to match training

        # 5) Predict
        prediction = model.predict(X_input)
        predicted_count = int(round(prediction[0]))

        return jsonify({"predicted_room_demand": predicted_count})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
