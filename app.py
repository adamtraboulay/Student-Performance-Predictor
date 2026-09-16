"""Streamlit app for predicting a student's final grade (G3)."""

import joblib
import pandas as pd
import streamlit as st

from src.preprocess import BINARY_COLS, CATEGORICAL_COLS, performance_category

MODEL_PATH = "models/model.pkl"
COLUMNS_PATH = "models/feature_columns.pkl"

st.set_page_config(page_title="Student Performance Predictor", page_icon="📊")
st.title("Student Performance Predictor")
st.write(
    "Estimate a student's final grade (G3, on a 0-20 scale) from their "
    "study habits, family background, and school life."
)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns


model, feature_columns = load_model()

with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home address", ["U", "R"], help="Urban / Rural")
        famsize = st.selectbox("Family size", ["LE3", "GT3"])
        Pstatus = st.selectbox("Parent status", ["T", "A"], help="Living together / Apart")
        Medu = st.slider("Mother's education (0-4)", 0, 4, 2)
        Fedu = st.slider("Father's education (0-4)", 0, 4, 2)
        Mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"])
        reason = st.selectbox("Reason for school choice", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        traveltime = st.slider("Home-to-school travel time (1-4)", 1, 4, 1)
        studytime = st.slider("Weekly study time (1-4)", 1, 4, 2)
        failures = st.slider("Past class failures", 0, 4, 0)

    with col2:
        schoolsup = st.selectbox("Extra school support", ["yes", "no"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Extra paid classes", ["yes", "no"])
        activities = st.selectbox("Extra-curricular activities", ["yes", "no"])
        nursery = st.selectbox("Attended nursery school", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet access at home", ["yes", "no"])
        romantic = st.selectbox("In a romantic relationship", ["yes", "no"])
        famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
        freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
        goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
        Dalc = st.slider("Workday alcohol consumption (1-5)", 1, 5, 1)
        Walc = st.slider("Weekend alcohol consumption (1-5)", 1, 5, 1)
        health = st.slider("Current health status (1-5)", 1, 5, 4)
        absences = st.slider("Number of school absences", 0, 93, 4)

    submitted = st.form_submit_button("Predict final grade")

if submitted:
    raw = pd.DataFrame([{
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
        "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences,
    }])

    for col in BINARY_COLS:
        raw[col] = raw[col].astype("category").cat.codes

    raw = pd.get_dummies(raw, columns=CATEGORICAL_COLS, drop_first=True)
    raw = raw.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(raw)[0]
    prediction = max(0, min(20, prediction))

    st.subheader(f"Predicted final grade: {prediction:.1f} / 20")
    st.write(f"Performance category: **{performance_category(prediction)}**")
