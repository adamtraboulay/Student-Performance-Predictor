"""Load and preprocess the UCI student performance dataset."""

import pandas as pd

TARGET = "G3"

BINARY_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
CATEGORICAL_COLS = ["Mjob", "Fjob", "reason", "guardian"]


def load_data(path="data/student_data.csv"):
    return pd.read_csv(path)


def preprocess(df):
    """Encode categorical columns and split into features/target.

    G1 and G2 are dropped from the features so the model predicts the
    final grade from behavioral/demographic data rather than just
    averaging the earlier grades.
    """
    df = df.copy()

    for col in BINARY_COLS:
        df[col] = df[col].astype("category").cat.codes

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    y = df[TARGET]
    X = df.drop(columns=[TARGET, "G1", "G2"])

    return X, y


def performance_category(score):
    if score < 10:
        return "Low"
    if score < 15:
        return "Medium"
    return "High"
