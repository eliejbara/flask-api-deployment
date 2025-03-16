# guest_prediction.py
import datetime
import joblib
import numpy as np
import pandas as pd  # <-- IMPORTANT: We need pandas to create DataFrame
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1) Load the trained model (which expects 11 features with specific names)
model = joblib.load("trained_model.pkl")

# 2) Define the exact column names in the same order you used in train_model.py
FEATURE_COLUMNS = [
    "avg_lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "day_of_week",
    "is_weekend",
    "month",
    "is_holiday_season",
    "days_out"
]

# 3) (Optional) day-of-week average dictionary for any fallback or example feature
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
    Return 11 numeric features that match FEATURE_COLUMNS in the same order:
      1) avg_lead_time
      2) stays_in_weekend_nights
      3) stays_in_week_nights
      4) adults
      5) children
      6) babies
      7) day_of_week
      8) is_weekend
      9) month
      10) is_holiday_season
      11) days_out
    """
    today = datetime.date.today()
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    # Calculate days until arrival
    days_out = (dt - today).days
    if days_out < 0:
        days_out = 0

    dow = dt.weekday()          # Monday=0, Sunday=6
    is_weekend = 1 if dow >= 5 else 0
    month = dt.month
    is_holiday_season = 1 if month in [7, 8, 12] else 0

    # Use dynamic_avgs for other numeric fields (or do your own logic)
    avg = dynamic_avgs.get(dow, {
        "lead_time": 100.0,
        "stays_in_weekend_nights": 1.0,
        "stays_in_week_nights": 2.0,
        "adults": 2.0,
        "children": 0.1,
        "babies": 0.0
    })

    # We'll use 'days_out' as both avg_lead_time and days_out
    lead_time = float(days_out)

    # Build the feature array in the exact same order
    features = [
        lead_time,                          # avg_lead_time
        avg["stays_in_weekend_nights"],    # stays_in_weekend_nights
        avg["stays_in_week_nights"],       # stays_in_week_nights
        avg["adults"],                     # adults
        avg["children"],                   # children
        avg["babies"],                     # babies
        dow,                               # day_of_week
        is_weekend,                        # is_weekend
        month,                             # month
        is_holiday_season,                 # is_holiday_season
        lead_time                          # days_out
    ]
    return features

@app.route('/predict', methods=['GET'])
def predict():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'Please provide a date parameter in YYYY-MM-DD format'}), 400

    try:
        # 1) Extract the numeric features
        feats = extract_features(date_str)

        # 2) Build a DataFrame with the same column names as training
        df_for_prediction = pd.DataFrame([feats], columns=FEATURE_COLUMNS)

        # 3) Predict using the DataFrame (removes the "X does not have valid feature names" warning)
        prediction = model.predict(df_for_prediction)[0]

        # 4) Round or int-cast if needed
        predicted_count = int(round(prediction))

        # 5) Return JSON
        return jsonify({
            'predicted_check_in_count': predicted_count,
            'used_features': feats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run your Flask app
    app.run(port=5000, debug=True)
