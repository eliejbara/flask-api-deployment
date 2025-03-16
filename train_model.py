# train_model.py
import pandas as pd
import numpy as np
import joblib
import calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_save_model():
    # 1. Load the cleaned dataset
    df = pd.read_csv("cleaned_hotel_bookings.csv")
    
    # 2. Convert one-hot month columns back to an integer month
    month_columns = [col for col in df.columns if col.startswith("arrival_date_month_")]
    if not month_columns:
        raise ValueError("No one-hot encoded month columns found (arrival_date_month_*).")

    # This picks whichever month column is '1' and extracts the month name
    df["arrival_date_month"] = df[month_columns].idxmax(axis=1).str.replace("arrival_date_month_", "", regex=True)
    
    # Map from month name (e.g., "January") to numeric month (1..12)
    month_mapping = {m: i for i, m in enumerate(calendar.month_name) if m}
    df["arrival_date_month"] = df["arrival_date_month"].map(month_mapping)
    df.dropna(subset=["arrival_date_month"], inplace=True)
    df["arrival_date_month"] = df["arrival_date_month"].astype(int)

    # 3. Rebuild the actual arrival_date from year, month, day
    df["arrival_date"] = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": df["arrival_date_month"],
        "day": df["arrival_date_day_of_month"]
    })

    # 4. Create additional columns
    df["day_of_week"] = df["arrival_date"].dt.weekday  # Monday=0, Sunday=6
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["month"] = df["arrival_date"].dt.month
    df["is_holiday_season"] = df["month"].apply(lambda x: 1 if x in [7, 8, 12] else 0)

    # 5. Aggregate daily data
    #    For each date, we sum or average the relevant features and get total check_in_count
    daily_features = df.groupby("arrival_date").agg({
        "lead_time": "mean",  # We'll call this avg_lead_time below
        "stays_in_weekend_nights": "sum",
        "stays_in_week_nights": "sum",
        "adults": "sum",
        "children": "sum",
        "babies": "sum",
        "day_of_week": "first",
        "is_weekend": "first",
        "month": "first",
        "is_holiday_season": "first"
    }).reset_index()

    check_in_counts = df.groupby("arrival_date").size().reset_index(name="check_in_count")
    daily_data = pd.merge(daily_features, check_in_counts, on="arrival_date")

    # Rename 'lead_time' to 'avg_lead_time' for clarity
    daily_data.rename(columns={"lead_time": "avg_lead_time"}, inplace=True)

    # 6. Create a 'days_out' feature
    #    Since we only have final booking data, we approximate 'days_out' by the average lead time
    #    In a real system, you'd use actual snapshots of how far away from the booking date we were.
    daily_data["days_out"] = daily_data["avg_lead_time"].round()

    # 7. Define features
    #    We'll now include 'days_out' at the end
    features = [
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
        "days_out"  # <-- New feature
    ]

    X = daily_data[features]
    y = daily_data["check_in_count"]

    # 8. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42
    )

    # 9. Train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 10. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Training Complete!")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.2f}")

    # 11. Save the trained model
    joblib.dump(model, "trained_model.pkl")
    print("Trained model saved as 'trained_model.pkl'.")

if __name__ == "__main__":
    train_and_save_model()
