from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import joblib
import os


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

    # 🔥 FORCE SAVE (WITH PRINT DEBUG)
    os.makedirs("models", exist_ok=True)

    print("Saving model...")
    joblib.dump(trained_models["gradient_boost"], "models/model.pkl")

    print("Saving columns...")
    joblib.dump(X.columns, "models/columns.pkl")

    print("Saved successfully!")

    return trained_models, X_test, y_test