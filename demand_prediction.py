from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://hotel-on-call.vercel.app"}})  # Allow frontend requests

print("Starting Flask API...")

model_file_path = "demand_model.pkl"

try:
    model = joblib.load(model_file_path)
    print(f"Model loaded successfully from {model_file_path}!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  

@app.route('/demand_prediction', methods=['OPTIONS'])
def handle_options():
    """Handles CORS preflight requests."""
    return '', 204

@app.route('/demand_prediction', methods=['GET', 'POST'])
def demand_prediction():
    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    try:
        year = int(request.args.get("year", 2025))
        month = int(request.args.get("month", 7))
        day_of_week = int(request.args.get("day_of_week", 4))
        is_weekend = int(request.args.get("is_weekend", 0))
        is_holiday_season = int(request.args.get("is_holiday_season", 0))
        avg_lead_time = float(request.args.get("avg_lead_time", 30))
        sum_previous_bookings = float(request.args.get("sum_previous_bookings", 5))
        avg_adr = float(request.args.get("avg_adr", 100))
        total_children = float(request.args.get("total_children", 2))

        # Build input features
        X_input = pd.DataFrame([{
            "year": year, "day_of_week": day_of_week, "is_weekend": is_weekend,
            "is_holiday_season": is_holiday_season, "avg_lead_time": avg_lead_time,
            "sum_previous_bookings": sum_previous_bookings, "avg_adr": avg_adr,
            "total_children": total_children
        }])

        # Predict
        prediction = model.predict(X_input)
        return jsonify({"predicted_room_demand": int(round(prediction[0]))})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
