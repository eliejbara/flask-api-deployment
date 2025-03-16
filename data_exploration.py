import pandas as pd
import matplotlib.pyplot as plt
import calendar

# Load the cleaned dataset
df = pd.read_csv("cleaned_hotel_bookings.csv")

# Display the columns to confirm the one-hot encoding
print("Columns in dataset:")
print(df.columns)

# Extract the month from the one-hot encoded columns:
# Find all columns that start with "arrival_date_month_"
month_columns = [col for col in df.columns if col.startswith("arrival_date_month_")]
if not month_columns:
    raise ValueError("No one-hot encoded arrival_date_month columns found!")

# For each row, the column with the maximum value (typically 1) indicates the month
df["arrival_date_month"] = df[month_columns].idxmax(axis=1).str.replace("arrival_date_month_", "", regex=True)

# Map month names (e.g., "August") to their corresponding numbers (e.g., 8)
month_mapping = {month: i for i, month in enumerate(calendar.month_name) if month}
df["arrival_date_month"] = df["arrival_date_month"].map(month_mapping)

# Drop rows where the month conversion failed
df = df.dropna(subset=["arrival_date_month"])
df["arrival_date_month"] = df["arrival_date_month"].astype(int)

# Create the 'arrival_date' column using the year, month, and day columns
df["arrival_date"] = pd.to_datetime({
    "year": df["arrival_date_year"],
    "month": df["arrival_date_month"],
    "day": df["arrival_date_day_of_month"]
})

# Aggregate daily check-in counts
check_in_counts = df.groupby("arrival_date").size().reset_index(name="check_in_count")

# Plot the distribution of daily check-in counts
plt.hist(check_in_counts["check_in_count"], bins=30, edgecolor='black')
plt.xlabel("Daily Check-in Count")
plt.ylabel("Frequency")
plt.title("Distribution of Daily Check-ins")
plt.show()
