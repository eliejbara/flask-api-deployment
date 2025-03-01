import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import datetime

# --- Step 1: Generate Sample Data ---
data = []
np.random.seed(42)
for i in range(60):
    # Generate a date (past 60 days)
    date = datetime.date.today() - datetime.timedelta(days=60 - i)
    day_of_week = date.weekday()  # Monday=0, Sunday=6
    # Simulate check-in count (base count with weekend boost)
    base_count = 50  
    if day_of_week >= 5:  # Saturday, Sunday
         count = base_count + np.random.randint(10, 20)
    else:
         count = base_count + np.random.randint(-5, 5)
    data.append([date, day_of_week, count])

df = pd.DataFrame(data, columns=['date', 'day_of_week', 'check_in_count'])

# --- Step 2: Train a Simple AI Model ---
# We use day_of_week as the feature for simplicity.
X = df[['day_of_week']]
y = df['check_in_count']

model = LinearRegression()
model.fit(X, y)

# --- Step 3: Set Up the Flask API ---
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Expect a 'date' query parameter (format: YYYY-MM-DD)
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'Please provide a date parameter in format YYYY-MM-DD'}), 400
    try:
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400
    # Calculate day of the week for the given date
    day_of_week = date_obj.weekday()
    # Predict check-in count
    prediction = model.predict(np.array([[day_of_week]]))
    predicted_count = int(round(prediction[0]))
    return jsonify({'predicted_check_in_count': predicted_count})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
