from src.data_preprocessing import load_data, preprocess
from src.feature_engineering import create_features
from src.train import train_model
from src.evaluate import evaluate

def main():
    df = load_data()
    df = preprocess(df)
    df = create_features(df)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    models, X_test, y_test = train_model(X, y)

    print("\n--- MODEL EVALUATION ---")
    for name, model in models.items():
        print(f"\n{name.upper()}")
        evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()