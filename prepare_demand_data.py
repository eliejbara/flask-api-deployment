# prepare_demand_data.py
import pandas as pd
import calendar

def create_demand_dataset():
    # 1) Load the cleaned dataset
    df = pd.read_csv("cleaned_hotel_bookings.csv")

    # 2) Convert one-hot month columns back to a single 'arrival_date_month' string
    month_columns = [col for col in df.columns if col.startswith("arrival_date_month_")]
    df["arrival_date_month"] = df[month_columns].idxmax(axis=1).str.replace("arrival_date_month_", "", regex=True)

    # Map month names to numeric (January=1, February=2, etc.)
    month_mapping = {m: i for i, m in enumerate(calendar.month_name) if m}
    df["arrival_date_month"] = df["arrival_date_month"].map(month_mapping)

    # 3) Create a proper 'arrival_date' column
    df["arrival_date"] = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": df["arrival_date_month"],
        "day": df["arrival_date_day_of_month"]
    })

    # 4) Aggregate daily data to form the target + additional features
    daily_agg = df.groupby("arrival_date").agg({
        "lead_time": "mean",
        "previous_bookings_not_canceled": "sum",
        "adr": "mean",
        "children": "sum"
    }).reset_index()

    daily_count = df.groupby("arrival_date").size().reset_index(name="total_bookings")

    daily_demand = pd.merge(daily_agg, daily_count, on="arrival_date")

    daily_demand.rename(columns={
        "lead_time": "avg_lead_time",
        "previous_bookings_not_canceled": "sum_previous_bookings",
        "adr": "avg_adr",
        "children": "total_children"
    }, inplace=True)

    # 5) Save the new dataset
    daily_demand.to_csv("daily_hotel_demand.csv", index=False)
    print("Created 'daily_hotel_demand.csv' with aggregated daily features!")

if __name__ == "__main__":
    create_demand_dataset()
