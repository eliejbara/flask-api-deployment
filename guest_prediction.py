# guest_prediction.py
import datetime
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model (which expects 11 features)
model = joblib.load("trained_model.pkl")

# Example day-of-week average dict (optional)
dynamic_avgs = {
    0: {"lead_time": 98.9, "stays_in_weekend_nights": 1.17, "stays_in_week_nights": 2.47, "adults": 1.83, "children": 0.10, "babies": 0.008},
    1: {"lead_time": 90.7, "stays_in_weekend_nights": 0.50, "stays_in_week_nights": 2.99, "adults": 1.78, "children": 0.10, "babies": 0.007},
    2: {"lead_time": 97.6, "stays_in_weekend_nights": 0.50, "stays_in_week_nights": 2.83, "adults": 1.80, "children": 0.08, "babies": 0.006},
    3: {"lead_time": 132.9, "stays_in_weekend_nights": 0.56, "stays_in_week_nights": 2.73, "adults": 1.87, "children": 0.09, "babies": 0.006},
    4: {"lead_time": 106.3, "stays_in_weekend_nights": 0.71, "stays_in_week_nights": 2.27, "adults": 1.88, "children": 0.10, "babies": 0.006},
    5: {"lead_time": 107.7, "stays_in_weekend_nights": 1.34, "stays_in_week_nights": 2.18, "adults": 1.92, "children": 0.12, "babies": 0.009},
    6: {"lead_time": 87.7,  "stays_in_weekend_nights": 1.78, "stays_in_week_nights": 2.07, "adults": 1.87, "children": 0.11, "babies": 0.010}
}

def extract_features(date_str):
    """
    Return 11 features to match train_model.py:
    1) lead_time (= days_out)
    2) stays_in_weekend_nights
    3) stays_in_week_nights
    4) adults
    5) children
    6) babies
    7) day_of_week
    8) is_weekend
    9) month
    10) is_holiday_season
    11) days_out (repeat the lead_time)
    """
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    today = datetime.date.today()

    days_out = (dt - today).days
    if days_out < 0:
        days_out = 0

    dow = dt.weekday()
    is_weekend = 1 if dow >= 5 else 0
    month = dt.month
    is_holiday_season = 1 if month in [7, 8, 12] else 0

    # You can still use dynamic_avgs for some typical numeric fields
    avg = dynamic_avgs.get(dow, {
        "lead_time": 100.0,
        "stays_in_weekend_nights": 1.0,
        "stays_in_week_nights": 2.0,
        "adults": 2.0,
        "children": 0.1,
        "babies": 0.0
    })

    # (1) lead_time => float(days_out)
    lead_time = float(days_out)

    # Build the feature vector with 11 entries
    features = [
        lead_time,                          # (1) avg_lead_time
        avg["stays_in_weekend_nights"],    # (2)
        avg["stays_in_week_nights"],       # (3)
        avg["adults"],                     # (4)
        avg["children"],                   # (5)
        avg["babies"],                     # (6)
        dow,                               # (7) day_of_week
        is_weekend,                        # (8)
        month,                             # (9)
        is_holiday_season,                 # (10)
        lead_time                          # (11) days_out
    ]

    return features

@app.route('/predict', methods=['GET'])
def predict():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'Please provide a date in YYYY-MM-DD format'}), 400

    try:
        feats = extract_features(date_str)
        # Model expects a 2D array
        prediction = model.predict([feats])
        predicted_count = int(round(prediction[0]))
        return jsonify({
            'predicted_check_in_count': predicted_count,
            'used_features': feats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
