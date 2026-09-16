"""Train and compare models for predicting a student's final grade (G3)."""

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess import load_data, preprocess


def main():
    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds),
        }
        print(f"{name}: MAE={results[name]['mae']:.3f}  R2={results[name]['r2']:.3f}")

    best_name = min(results, key=lambda n: results[n]["mae"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(list(X.columns), "models/feature_columns.pkl")
    print("Saved models/model.pkl and models/feature_columns.pkl")


if __name__ == "__main__":
    main()
