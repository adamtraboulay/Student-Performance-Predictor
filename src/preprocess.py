# loads the csv and gets it ready for the model
# basically just turning all the yes/no and text columns into numbers
# since sklearn can't work with strings

import numpy as np
import pandas as pd

TARGET = "G3"  # this is the final grade, what we're trying to predict

# fixed value -> code maps for the yes/no / 2-option columns. Has to be a
# fixed map (not pandas' .astype("category").cat.codes) because cat.codes
# derives its categories from whatever rows are present - on a single-row
# dataframe (like a live prediction) every column only has ONE distinct
# value, so cat.codes always encodes it as 0 no matter what the value
# actually was. Learned this the hard way when the app was silently
# ignoring sex/internet/schoolsup/etc at prediction time.
BINARY_MAPS = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},
    "famsize": {"GT3": 0, "LE3": 1},
    "Pstatus": {"A": 0, "T": 1},
    "schoolsup": {"no": 0, "yes": 1},
    "famsup": {"no": 0, "yes": 1},
    "paid": {"no": 0, "yes": 1},
    "activities": {"no": 0, "yes": 1},
    "nursery": {"no": 0, "yes": 1},
    "higher": {"no": 0, "yes": 1},
    "internet": {"no": 0, "yes": 1},
    "romantic": {"no": 0, "yes": 1},
}
BINARY_COLS = list(BINARY_MAPS)

# same issue applies to pd.get_dummies: with drop_first=True, a single-row
# dataframe only ever has ONE category present, so it gets treated as "the
# first" and dropped every time, regardless of which value it actually is.
# Fixing the category list up front makes get_dummies always emit the same
# columns whether it's encoding 395 rows or 1.
CATEGORICAL_OPTIONS = {
    "Mjob": ["at_home", "health", "other", "services", "teacher"],
    "Fjob": ["at_home", "health", "other", "services", "teacher"],
    "reason": ["course", "home", "other", "reputation"],
    "guardian": ["father", "mother", "other"],
}
CATEGORICAL_COLS = list(CATEGORICAL_OPTIONS)

# allowed values per field, for validating input coming from the web app
ENUM_FIELDS = {col: list(mapping.keys()) for col, mapping in BINARY_MAPS.items()}
ENUM_FIELDS.update(CATEGORICAL_OPTIONS)

# (min, max) per numeric field, matches the ranges in the dataset/original form
RANGE_FIELDS = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 93),
}


def load_data(path="data/student_data.csv"):
    return pd.read_csv(path)


def encode_common(df):
    # make a copy so we don't mess up the original dataframe
    df = df.copy()

    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping)

    for col, options in CATEGORICAL_OPTIONS.items():
        df[col] = pd.Categorical(df[col], categories=options)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # absences is heavily right-skewed (most students have < 10, a few have
    # 50-75), log1p pulls those outliers in so they don't dominate
    df["absences"] = np.log1p(df["absences"])

    return df


def build_blind_features(df):
    """Features you'd have without knowing the student's recent grades."""
    df = encode_common(df)
    y = df[TARGET]
    X = df.drop(columns=[TARGET, "G1", "G2"])
    return X, y


def build_full_features(df):
    """Same as blind, but keeps G1/G2 (recent period grades) - these are
    by far the strongest predictor of G3."""
    df = encode_common(df)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def performance_category(score):
    # just some rough cutoffs to turn the number grade into a category
    if score < 10:
        return "Low"
    if score < 15:
        return "Medium"
    return "High"
