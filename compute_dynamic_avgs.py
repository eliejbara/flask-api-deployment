# guest_prediction.py
import datetime
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model (ensure this is the model from train_model.py)
model = joblib.load("trained_model.pkl")

# Dynamic averages for non-date features (update with your computed values)
dynamic_avgs = {
    0: {"lead_time": 98.91, "stays_in_weekend_nights": 1.1763, "stays_in_week_nights": 2.4693, "adults": 1.8312, "children": 0.1030, "babies": 0.00846},
    1: {"lead_time": 90.7438, "stays_in_weekend_nights": 0.50623, "stays_in_week_nights": 2.99282, "adults": 1.7889, "children": 0.10217, "babies": 0.00779},
    2: {"lead_time": 97.6442, "stays_in_weekend_nights": 0.50003, "stays_in_week_nights": 2.83197, "adults": 1.80468, "children": 0.08897, "babies": 0.00632},
    3: {"lead_time": 132.9514, "stays_in_weekend_nights": 0.56293, "stays_in_week_nights": 2.73252, "adults": 1.87319, "children": 0.09214, "babies": 0.00611},
    4: {"lead_time": 106.3537, "stays_in_weekend_nights": 0.71105, "stays_in_week_nights": 2.27059, "adults": 1.88955, "children": 0.10595, "babies": 0.00683},
    5: {"lead_time": 107.7431, "stays_in_weekend_nights": 1.34241, "stays_in_week_nights": 2.18017, "adults": 1.92722, "children": 0.12633, "babies": 0.00976},
    6: {"lead_time": 87.7703, "stays_in_weekend_nights": 1.78051, "stays_in_week_nights": 2.07198, "adults": 1.87901, "children": 0.11110, "babies": 0.01093}
}

def extract_features(date_str):
    """
    Given a date string in YYYY-MM-DD format, extract a 10-feature vector:
      [lead_time, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies,
       day_of_week, is_weekend, month, is_holiday_season]
    """
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    dow = dt.weekday()  # Monday=0, Sunday=6
    is_weekend = 1 if dow >= 5 else 0
    month = dt.month
    # Define holiday season: example, July, August, December are high demand
    is_holiday_season = 1 if month in [7, 8, 12] else 0

    # Get dynamic averages for this day-of-week
    avg = dynamic_avgs.get(dow, {
        "lead_time": 100.0,
        "stays_in_weekend_nights": 1.0,
        "stays_in_week_nights": 2.0,
        "adults": 2.0,
        "children": 0.1,
        "babies": 0.0
    })

    features = [
        avg["lead_time"],
        avg["stays_in_weekend_nights"],
        avg["stays_in_week_nights"],
        avg["adults"],
        avg["children"],
        avg["babies"],
        dow,
        is_weekend,
        month,
        is_holiday_season
    ]
    return features

@app.route('/predict', methods=['GET'])
def predict():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'Please provide a date parameter in YYYY-MM-DD format'}), 400
    try:
        features = extract_features(date_str)
        prediction = model.predict(np.array([features]))
        predicted_count = int(round(prediction[0]))
        print(f"Date: {date_str}, Features: {features}, Predicted: {predicted_count}")
        return jsonify({'predicted_check_in_count': predicted_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
