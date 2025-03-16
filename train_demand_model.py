# train_demand_model.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_demand_model():
    # 1) Load the daily demand dataset
    df = pd.read_csv("daily_hotel_demand.csv")

    # 2) Convert 'arrival_date' to datetime
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])

    # 3) Extract time-based features
    df["year"] = df["arrival_date"].dt.year
    df["month"] = df["arrival_date"].dt.month
    df["day_of_week"] = df["arrival_date"].dt.weekday  # Monday=0, Sunday=6
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # 4) Create a holiday season flag
    df["is_holiday_season"] = df["month"].apply(lambda x: 1 if x in [7, 8, 12] else 0)

    # 5) Force the 'month' column to have categories 1..12
    all_months = list(range(1, 13))
    df["month"] = pd.Categorical(df["month"], categories=all_months)

    # 6) One-hot encode the 'month' column
    df = pd.get_dummies(df, columns=["month"], prefix="month", drop_first=True)
    # => month_2..month_12

    # 7) Define the final feature columns in the EXACT order we want
    #    This must match what we use in demand_prediction.py
    dummy_month_cols = [f"month_{m}" for m in range(2, 13)]
    FEATURES = [
        "year",         # numeric
        "day_of_week",  # numeric
        "is_weekend",   # numeric (0/1)
        "is_holiday_season",   # numeric (0/1)
        "avg_lead_time",       # from daily_demand
        "sum_previous_bookings",  # from daily_demand
        "avg_adr",             # from daily_demand
        "total_children"       # from daily_demand
    ] + dummy_month_cols

    # 8) Our target
    y = df["total_bookings"]

    # 9) X is everything except arrival_date + target
    X = df.drop(columns=["arrival_date", "total_bookings"])

    # 10) Reorder X to match FEATURES
    #     This ensures scikit-learn sees columns in a stable order
    X = X[FEATURES]

    # 11) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 12) Train a Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 13) Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Training Complete!")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.2f}")

    # 14) Save the model
    joblib.dump(model, "demand_model.pkl")
    print("Demand model saved as 'demand_model.pkl'.")

    # 15) Show Feature Importances
    feat_importances = model.feature_importances_
    feature_names = FEATURES

    plt.figure(figsize=(10,6))
    plt.barh(feature_names, feat_importances, color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance in RandomForest for Room Demand")
    plt.show()

if __name__ == "__main__":
    train_demand_model()
