import pandas as pd
import numpy as np
import os

# Get the main project folder automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset location
file_path = os.path.join(BASE_DIR, "data", "Online Retail.xlsx")

print("Dataset path:", file_path)
print("File exists:", os.path.exists(file_path))

# Load dataset
df = pd.read_excel(file_path)

# Display basic dataset information
print("\n===== DATA COLLECTION SUCCESSFUL =====")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\n===== INITIAL INSPECTION COMPLETED =====")


# ==========================================
# STEP 2: DATA QUALITY INVESTIGATION
# ==========================================

print("\n===== DATA QUALITY INVESTIGATION =====")

# 1. Check negative quantities
negative_quantity = df[df["Quantity"] < 0]

print("\nNegative Quantity Records:")
print(len(negative_quantity))

# 2. Check zero and negative unit prices
zero_price = df[df["UnitPrice"] == 0]
negative_price = df[df["UnitPrice"] < 0]

print("\nZero Unit Price Records:")
print(len(zero_price))

print("\nNegative Unit Price Records:")
print(len(negative_price))

# 3. Identify cancelled invoices
cancelled_invoices = df[
    df["InvoiceNo"].astype(str).str.startswith("C")
]

print("\nCancelled Invoice Records:")
print(len(cancelled_invoices))

# 4. Check missing descriptions
print("\nMissing Product Descriptions:")
print(df["Description"].isnull().sum())

# 5. Check date range
print("\nInvoice Date Range:")
print("Starting Date:", df["InvoiceDate"].min())
print("Ending Date:", df["InvoiceDate"].max())

# 6. Detect outliers using IQR method
print("\n===== OUTLIER DETECTION =====")

for column in ["Quantity", "UnitPrice"]:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\nColumn: {column}")
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Outlier Count:", len(outliers))

print("\n===== INVESTIGATION COMPLETED =====")


# ==========================================
# STEP 3: DATA CLEANING
# ==========================================

print("\n===== DATA CLEANING STARTED =====")

# Create output folder if it does not exist
os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)

# Make a working copy (original dataset remains unchanged)
clean_df = df.copy()

# 1. Remove duplicate records
before_duplicates = len(clean_df)

clean_df = clean_df.drop_duplicates()

duplicates_removed = before_duplicates - len(clean_df)

print("\nDuplicates Removed:", duplicates_removed)

# 2. Separate cancelled invoices
cancelled_df = clean_df[
    clean_df["InvoiceNo"].astype(str).str.startswith("C")
].copy()

# Keep only non-cancelled invoices
clean_df = clean_df[
    ~clean_df["InvoiceNo"].astype(str).str.startswith("C")
].copy()

print("Cancelled Records Separated:", len(cancelled_df))

# 3. Separate possible product returns
returns_df = clean_df[clean_df["Quantity"] < 0].copy()

# Keep positive quantity records for sales analysis
clean_df = clean_df[clean_df["Quantity"] > 0].copy()

print("Potential Return Records Separated:", len(returns_df))

# 4. Remove missing product descriptions
before_missing = len(clean_df)

clean_df = clean_df.dropna(subset=["Description"])

missing_removed = before_missing - len(clean_df)

print("Missing Descriptions Removed:", missing_removed)

# 5. Separate unusual price records
price_exceptions = clean_df[
    clean_df["UnitPrice"] <= 0
].copy()

# Retain only positive prices for standard sales analysis
clean_df = clean_df[clean_df["UnitPrice"] > 0].copy()

print("Zero/Negative Price Records Separated:", len(price_exceptions))

# 6. Calculate transaction value
clean_df["SalesValue"] = (
    clean_df["Quantity"] * clean_df["UnitPrice"]
)

print("\nCleaned Dataset Shape:", clean_df.shape)

# ==========================================
# STEP 4: OUTLIER IDENTIFICATION
# ==========================================

print("\n===== OUTLIER FLAGGING =====")

for column in ["Quantity", "UnitPrice"]:

    Q1 = clean_df[column].quantile(0.25)
    Q3 = clean_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    flag_column = column + "_Outlier"

    clean_df[flag_column] = (
        (clean_df[column] < lower_limit) |
        (clean_df[column] > upper_limit)
    )

    print(f"\n{column} Outliers:",
          clean_df[flag_column].sum())

# Outliers are flagged, not automatically deleted.

# ==========================================
# STEP 5: NORMALIZATION
# ==========================================

from sklearn.preprocessing import MinMaxScaler

print("\n===== DATA NORMALIZATION =====")

scaler = MinMaxScaler()

columns_to_scale = ["Quantity", "UnitPrice"]

scaled_values = scaler.fit_transform(
    clean_df[columns_to_scale]
)

clean_df["Quantity_Normalized"] = scaled_values[:, 0]
clean_df["UnitPrice_Normalized"] = scaled_values[:, 1]

print("\nNormalization Completed!")

print(clean_df[
    ["Quantity_Normalized", "UnitPrice_Normalized"]
].describe())

# ==========================================
# STEP 6: SAVE PROCESSED DATASETS
# ==========================================

output_folder = os.path.join(BASE_DIR, "output")

clean_df.to_csv(
    os.path.join(output_folder, "cleaned_logistics_data.csv"),
    index=False
)

returns_df.to_csv(
    os.path.join(output_folder, "potential_returns.csv"),
    index=False
)

cancelled_df.to_csv(
    os.path.join(output_folder, "cancelled_invoices.csv"),
    index=False
)

price_exceptions.to_csv(
    os.path.join(output_folder, "price_exceptions.csv"),
    index=False
)

print("\n===== FILES SAVED SUCCESSFULLY =====")

print("Cleaned sales data saved.")
print("Potential returns saved.")
print("Cancelled invoices saved.")
print("Price exceptions saved.")

print("\nFinal Dataset Shape:", clean_df.shape)

print("\n===== PREPROCESSING COMPLETED =====")


# ==========================================
# STEP 7: FINAL DATA VALIDATION
# ==========================================

print("\n===== FINAL DATA VALIDATION =====")

# Check remaining missing values
print("\nRemaining Missing Values:")
print(clean_df.isnull().sum())

# Check remaining duplicate records
print("\nRemaining Duplicate Records:")
print(clean_df.duplicated().sum())

# Check numerical value ranges
print("\nQuantity Range:")
print(clean_df["Quantity"].min(),
      "to", clean_df["Quantity"].max())

print("\nUnit Price Range:")
print(clean_df["UnitPrice"].min(),
      "to", clean_df["UnitPrice"].max())

# Check normalized value ranges
print("\nNormalized Quantity Range:")
print(clean_df["Quantity_Normalized"].min(),
      "to", clean_df["Quantity_Normalized"].max())

print("\nNormalized Unit Price Range:")
print(clean_df["UnitPrice_Normalized"].min(),
      "to", clean_df["UnitPrice_Normalized"].max())

# Check remaining customer ID values
print("\nMissing Customer IDs:")
print(clean_df["CustomerID"].isnull().sum())

# Final validation status
print("\n===== VALIDATION SUMMARY =====")

print("Total Records:", len(clean_df))
print("Total Columns:", len(clean_df.columns))
print("Duplicates Remaining:", clean_df.duplicated().sum())
print("Missing Descriptions:",
      clean_df["Description"].isnull().sum())

print("\n===== VALIDATION COMPLETED =====")
