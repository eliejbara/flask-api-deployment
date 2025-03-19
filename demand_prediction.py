import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# CORS configuration - Allow frontend from specific origin (Vercel)
CORS(app, resources={r"/*": {"origins": "https://hotel-on-call.vercel.app"}})

print("Starting Flask API...")

# Get the model file path (ensure the model file is in the right location)
model_file_path = "demand_model.pkl"  # Adjust this if the model file is in a different directory

try:
    # Try loading the model
    model = joblib.load(model_file_path)
    print(f"✅ Model loaded successfully from {model_file_path}!")
except Exception as e:
    # If model loading fails, log the error
    print(f"❌ Error loading model: {e}")
    model = None  # Set model to None if loading fails

@app.route('/demand_prediction', methods=['OPTIONS'])
def handle_options():
    """Handles CORS preflight requests"""
    return '', 204  # 204 No Content for preflight request

@app.route('/demand_prediction', methods=['GET', 'POST'])
def demand_prediction():
    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    try:
        # Get input parameters (with default values if not provided)
        year = int(request.args.get("year", 2025))
        month = int(request.args.get("month", 7))
        day_of_week = int(request.args.get("day_of_week", 4))
        is_weekend = int(request.args.get("is_weekend", 0))
        is_holiday_season = int(request.args.get("is_holiday_season", 0))
        avg_lead_time = float(request.args.get("avg_lead_time", 30))
        sum_previous_bookings = float(request.args.get("sum_previous_bookings", 5))
        avg_adr = float(request.args.get("avg_adr", 100))
        total_children = float(request.args.get("total_children", 2))

        # Prepare the feature dictionary with default values
        row_dict = {
            "year": year, "day_of_week": day_of_week, "is_weekend": is_weekend,
            "is_holiday_season": is_holiday_season, "avg_lead_time": avg_lead_time,
            "sum_previous_bookings": sum_previous_bookings, "avg_adr": avg_adr,
            "total_children": total_children
        }

        # Dummy month columns (all set to 0 initially)
        DUMMY_MONTH_COLS = [f"month_{m}" for m in range(2, 13)]
        for m_col in DUMMY_MONTH_COLS:
            row_dict[m_col] = 0

        # Set the correct month column to 1
        month_col = f"month_{month}"
        if month_col in row_dict:
            row_dict[month_col] = 1

        # Create a DataFrame ensuring the columns are in the correct order
        FEATURES = [
            "year", "day_of_week", "is_weekend", "is_holiday_season", "avg_lead_time",
            "sum_previous_bookings", "avg_adr", "total_children"
        ] + DUMMY_MONTH_COLS

        X_input = pd.DataFrame([row_dict])
        X_input = X_input[FEATURES]

        # Make the prediction
        prediction = model.predict(X_input)
        predicted_count = int(round(prediction[0]))

        # Return the prediction as a JSON response
        return jsonify({"predicted_room_demand": predicted_count})
    
    except Exception as e:
        # Return an error if something goes wrong in prediction
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Set the port from the environment or use 5000 by default
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
