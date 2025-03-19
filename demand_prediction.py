import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

print("Starting Flask API...")

# Get the absolute path to the model file (adjust the path as needed)
model_file_path = "demand_model.pkl"  # Make sure this path is correct

# Try to load the trained model and log the status
try:
    model = joblib.load(model_file_path)
    print(f"Model loaded successfully from {model_file_path}!")
except Exception as e:
    print(f"Error loading model from {model_file_path}: {e}")

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

@app.route('/api/demand_prediction', methods=['GET'])
def predict_demand():
    try:
        # Parse input parameters
        year = int(request.args.get("year", 2025))
        month = int(request.args.get("month", 7))
        day_of_week = int(request.args.get("day_of_week", 4))
        is_weekend = int(request.args.get("is_weekend", 0))
        is_holiday_season = int(request.args.get("is_holiday_season", 0))
        avg_lead_time = float(request.args.get("avg_lead_time", 30))
        sum_previous_bookings = float(request.args.get("sum_previous_bookings", 5))
        avg_adr = float(request.args.get("avg_adr", 100))
        total_children = float(request.args.get("total_children", 2))

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

        for m_col in DUMMY_MONTH_COLS:
            row_dict[m_col] = 0

        month_col = f"month_{month}"
        if month_col in row_dict:
            row_dict[
