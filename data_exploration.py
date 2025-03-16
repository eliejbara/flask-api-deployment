import pandas as pd

# Load the dataset
df = pd.read_csv('ml/hotel_bookings.csv')

# Display the first few rows
print(df.head())

# Display dataset shape (number of rows and columns)
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Display column names
print("Columns in dataset:", df.columns)
