
import pandas as pd

def load_data():
    return pd.read_csv("data/telco.csv")

def preprocess(df):
    print("Columns:", df.columns)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    print("Normalized Columns:", df.columns)

    # Drop ID
    if "customer_id" in df.columns:
        df = df.drop("customer_id", axis=1)

    # Target
    if "churn_label" in df.columns:
        df["churn"] = df["churn_label"].map({"Yes": 1, "No": 0})
        df = df.drop("churn_label", axis=1)
    else:
        raise ValueError("Target column not found")

    # Remove leakage
    drop_cols = [
        "churn_category",
        "churn_reason",
        "customer_status",
        "churn_score"
    ]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    leakage_cols = [
        "satisfaction_score",
        "cltv",
        "total_revenue"
    ]

    for col in leakage_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Convert numeric
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    df = df.dropna()

    # Encoding
    df = pd.get_dummies(df, drop_first=True)

    return df