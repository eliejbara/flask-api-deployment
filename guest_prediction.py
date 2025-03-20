from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import pandas as pd

app = Flask(__name__)

# Enable CORS for Vercel frontend
CORS(app, resources={r"/*": {"origins": "https://hotel-on-call.vercel.app", "supports_credentials": True}})

print("Starting Flask API...")

# Load the model
model_file_path = "demand_model.pkl"
try:
    model = joblib.load(model_file_path)
    print(f"✅ Model loaded successfully from {model_file_path}!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Preflight request handler (OPTIONS request)
@app.route('/api/predict-demand', methods=['OPTIONS'])
def handle_preflight():
    response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "https://hotel-on-call.vercel.app")
    response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response, 200

# Endpoint to handle prediction
@app.route('/api/predict-demand', methods=['GET'])
def predict_demand():
    if model is None:
        response = jsonify({"error": "Model failed to load"})
        response.headers.add("Access-Control-Allow-Origin", "https://hotel-on-call.vercel.app")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response, 500

    try:
        # Extract parameters with default values
        params = ["year", "month", "day_of_week", "is_weekend", "is_holiday_season",
                  "avg_lead_time", "sum_previous_bookings", "avg_adr", "total_children"]
        data = {}

        for param in params:
            value = request.args.get(param)
            if value is None:
                return jsonify({"error": f"Missing required parameter: {param}"}), 400
            data[param] = float(value)

        print("Data received:", data)  # Log the input data

        # Convert necessary values to integers
        data["year"] = int(data["year"])
        data["month"] = int(data["month"])
        data["day_of_week"] = int(data["day_of_week"])

        # Prepare feature dictionary
        row_dict = {key: data[key] for key in params if key != "month"}

        # Handle month dummy variables
        DUMMY_MONTH_COLS = [f"month_{m}" for m in range(1, 13)]
        for m_col in DUMMY_MONTH_COLS:
            row_dict[m_col] = 1 if m_col == f"month_{data['month']}" else 0

        print("Feature dictionary:", row_dict)  # Log the features before prediction

        # Feature order
        FEATURES = list(row_dict.keys())
        X_input = pd.DataFrame([row_dict])[FEATURES]

        # Make prediction
        prediction = model.predict(X_input)
        predicted_count = int(round(prediction[0]))

        print("Prediction result:", predicted_count)  # Log prediction result

        # Response with headers
        response = jsonify({"predicted_room_demand": predicted_count})
        response.headers.add("Access-Control-Allow-Origin", "https://hotel-on-call.vercel.app")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "https://hotel-on-call.vercel.app")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response, 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 
