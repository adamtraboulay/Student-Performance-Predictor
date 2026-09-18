# trains two model variants:
#   - "blind": predicts G3 from behavior/demographics only (no prior grades)
#   - "full": also gets G1/G2 (recent period grades), much more predictive
#
# for each variant, tries a few model types (with a small hyperparameter
# search each) and keeps whichever gets the best cross-validated R2, since
# with only ~400 rows a single train/test split is pretty noisy

import joblib
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import build_blind_features, build_full_features, load_data

CV = KFold(n_splits=5, shuffle=True, random_state=42)

# (estimator, param grid) per candidate - ridge needs scaling since it's
# regularized, trees don't care about feature scale
CANDIDATE_GRIDS = {
    "ridge": (
        Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        {"model__alpha": [1, 5, 10, 20, 50]},
    ),
    "random_forest": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [200, 400],
            "max_depth": [None, 6, 8],
            "max_features": ["sqrt", 0.5],
            "min_samples_leaf": [1, 3],
        },
    ),
    "gradient_boosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
            "subsample": [0.8, 1.0],
        },
    ),
}


def select_best(X_train, y_train):
    best_name, best_search = None, None
    for name, (estimator, param_grid) in CANDIDATE_GRIDS.items():
        search = GridSearchCV(estimator, param_grid, cv=CV, scoring="r2", n_jobs=-1)
        search.fit(X_train, y_train)
        print(f"    {name}: CV R2={search.best_score_:.3f}  params={search.best_params_}")
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search
    return best_name, best_search


def train_variant(label, X, y, model_path, columns_path):
    print(f"\n=== {label} ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_name, search = select_best(X_train, y_train)

    # honest performance estimate: this model only saw X_train during
    # training/selection, X_test is untouched
    test_preds = search.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f"  best: {best_name}  test R2={test_r2:.3f}  test MAE={test_mae:.3f}")

    # now refit the winning model (same hyperparams) on ALL the data before
    # saving it, since more training data should only help the shipped model
    final_model = clone(search.best_estimator_)
    final_model.fit(X, y)

    joblib.dump(final_model, model_path)
    joblib.dump(list(X.columns), columns_path)
    print(f"  saved {model_path} and {columns_path}")

    return test_r2, test_mae


def main():
    df = load_data()

    X_blind, y_blind = build_blind_features(df)
    X_full, y_full = build_full_features(df)

    train_variant(
        "Blind model (no G1/G2)",
        X_blind, y_blind,
        "models/model_blind.pkl", "models/feature_columns_blind.pkl",
    )
    train_variant(
        "Full model (with G1/G2)",
        X_full, y_full,
        "models/model_full.pkl", "models/feature_columns_full.pkl",
    )


if __name__ == "__main__":
    main()
