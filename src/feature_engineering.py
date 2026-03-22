def create_features(df):
    if "tenure_in_months" in df.columns:
        df["tenure_group"] = df["tenure_in_months"] // 12
    return df


