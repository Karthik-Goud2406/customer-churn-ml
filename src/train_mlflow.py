from ast import main

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os
import pandas as pd


def show_feature_importance(model, X):
    importances = model.feature_importances_
    features = X.columns

    df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nTop Features:\n", df.head(10))


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        "random_forest": RandomForestClassifier(),
        "gradient_boost": GradientBoostingClassifier(),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss"
        )
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(trained_models["gradient_boost"], "models/model.pkl")

    # Show feature importance
    show_feature_importance(trained_models["gradient_boost"], X)

    return trained_models, X_test, y_test



def main():
    ...
    print("Logged to MLflow")

if __name__ == "__main__":
    main()