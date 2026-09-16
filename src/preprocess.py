# loads the csv and gets it ready for the model
# basically just turning all the yes/no and text columns into numbers
# since sklearn can't work with strings

import pandas as pd

TARGET = "G3"  # this is the final grade, what we're trying to predict

# columns that are just yes/no or have 2 options, easy to encode
BINARY_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
# columns with more than 2 categories, need one hot encoding for these
CATEGORICAL_COLS = ["Mjob", "Fjob", "reason", "guardian"]


def load_data(path="data/student_data.csv"):
    return pd.read_csv(path)


def preprocess(df):
    # make a copy so we don't mess up the original dataframe
    df = df.copy()

    # cat.codes turns each category into a number (0, 1, 2...)
    for col in BINARY_COLS:
        df[col] = df[col].astype("category").cat.codes

    # get_dummies does one hot encoding for the columns with more categories
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    y = df[TARGET]
    # dropping G1 and G2 too because otherwise the model just learns to
    # average them which felt like cheating / not the point of the project
    X = df.drop(columns=[TARGET, "G1", "G2"])

    return X, y


def performance_category(score):
    # just some rough cutoffs to turn the number grade into a category
    if score < 10:
        return "Low"
    if score < 15:
        return "Medium"
    return "High"
