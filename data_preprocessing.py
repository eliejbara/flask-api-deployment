import pandas as pd

# Load the dataset
df = pd.read_csv("hotel_bookings.csv")

# ✅ Drop irrelevant columns
df = df.drop(columns=["reservation_status", "reservation_status_date", "company", "agent"])

# ✅ Handle missing values
df["children"] = df["children"].fillna(0)  # Fixed the warning
df.dropna(inplace=True)  # Drop remaining missing rows

# ✅ Convert categorical features to numeric (One-Hot Encoding)
df = pd.get_dummies(df, columns=["hotel", "meal", "market_segment", "distribution_channel",
                                 "reserved_room_type", "assigned_room_type", "deposit_type",
                                 "customer_type", "arrival_date_month"], drop_first=True)

# ✅ Save the cleaned dataset
df.to_csv("cleaned_hotel_bookings.csv", index=False)

# Print results
print("Data preprocessing completed!")  # Fixed UnicodeEncodeError
print(f"Dataset now contains {df.shape[0]} rows and {df.shape[1]} columns.")
