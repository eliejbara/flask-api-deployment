import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for frontend hosted on Vercel (specific domain)

CORS(app, resources={r"/*": {"origins": ["https://hotel-on-call.vercel.app", "http://localhost:3000"]}}, supports_credentials=True)

print("Starting Flask API...")

# Load the model
model_file_path = "demand_model.pkl"  # Adjust path if needed

try:
    model = joblib.load(model_file_path)
    print(f"✅ Model loaded successfully from {model_file_path}!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Endpoint to handle prediction
@app.route('/demand_prediction', methods=['GET'])
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

        # Prepare feature dictionary
        row_dict = {
            "year": year, "day_of_week": day_of_week, "is_weekend": is_weekend,
            "is_holiday_season": is_holiday_season, "avg_lead_time": avg_lead_time,
            "sum_previous_bookings": sum_previous_bookings, "avg_adr": avg_adr,
            "total_children": total_children
        }

        # Handle month dummy variables
        DUMMY_MONTH_COLS = [f"month_{m}" for m in range(2, 13)]
        for m_col in DUMMY_MONTH_COLS:
            row_dict[m_col] = 0
        
        # Set the month column
        month_col = f"month_{month}"
        if month_col in row_dict:
            row_dict[month_col] = 1

        # Feature order
        FEATURES = [
            "year", "day_of_week", "is_weekend", "is_holiday_season", "avg_lead_time",
            "sum_previous_bookings", "avg_adr", "total_children"
        ] + DUMMY_MONTH_COLS

        X_input = pd.DataFrame([row_dict])
        X_input = X_input[FEATURES]

        # Make prediction
        prediction = model.predict(X_input)
        predicted_count = int(round(prediction[0]))

        # Return prediction result
        return jsonify({"predicted_room_demand": predicted_count})

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use the environment port or default to 5000
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
