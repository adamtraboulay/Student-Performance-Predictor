# streamlit app, this is just the ui part
# it takes all the inputs, runs it through the same preprocessing as
# training, then loads the saved model and predicts

import joblib
import pandas as pd
import streamlit as st

from src.preprocess import BINARY_COLS, CATEGORICAL_COLS, performance_category

MODEL_PATH = "models/model.pkl"
COLUMNS_PATH = "models/feature_columns.pkl"

st.title("Student Performance Predictor")
st.write("fill in the info below and it'll guess the final grade (G3, out of 20)")
st.caption("school project, accuracy isn't great but it works :)")


# cache_resource so it doesn't reload the model every single time
# someone clicks the button, was slow without this
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns


model, feature_columns = load_model()

# putting everything in a form so it only reruns once you hit submit
# instead of every time you touch a slider
with st.form("student_form"):
    school = st.selectbox("School (GP or MS)", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.slider("Age", 15, 22, 17)
    address = st.selectbox("Home address (U = urban, R = rural)", ["U", "R"])
    famsize = st.selectbox("Family size (LE3 = 3 or less, GT3 = more than 3)", ["LE3", "GT3"])
    Pstatus = st.selectbox("Parents living together? (T = together, A = apart)", ["T", "A"])
    Medu = st.slider("Mom's education level (0 = none, 4 = higher ed)", 0, 4, 2)
    Fedu = st.slider("Dad's education level (0 = none, 4 = higher ed)", 0, 4, 2)
    Mjob = st.selectbox("Mom's job", ["teacher", "health", "services", "at_home", "other"])
    Fjob = st.selectbox("Dad's job", ["teacher", "health", "services", "at_home", "other"])
    reason = st.selectbox("Why they picked this school", ["home", "reputation", "course", "other"])
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])
    traveltime = st.slider("Travel time to school (1 = short, 4 = long)", 1, 4, 1)
    studytime = st.slider("Weekly study time (1 = <2hrs, 4 = 10+hrs)", 1, 4, 2)
    failures = st.slider("Past class failures", 0, 4, 0)
    schoolsup = st.selectbox("Extra school support?", ["yes", "no"])
    famsup = st.selectbox("Family support with school stuff?", ["yes", "no"])
    paid = st.selectbox("Extra paid classes?", ["yes", "no"])
    activities = st.selectbox("Does extra-curriculars?", ["yes", "no"])
    nursery = st.selectbox("Went to nursery school?", ["yes", "no"])
    higher = st.selectbox("Wants to go to uni?", ["yes", "no"])
    internet = st.selectbox("Has internet at home?", ["yes", "no"])
    romantic = st.selectbox("In a relationship?", ["yes", "no"])
    famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
    freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
    goout = st.slider("Goes out with friends (1-5)", 1, 5, 3)
    Dalc = st.slider("Alcohol on weekdays (1-5)", 1, 5, 1)
    Walc = st.slider("Alcohol on weekends (1-5)", 1, 5, 1)
    health = st.slider("Health (1 = bad, 5 = great)", 1, 5, 4)
    absences = st.slider("Number of absences", 0, 93, 4)

    submitted = st.form_submit_button("Predict my grade")

if submitted:
    # need to build this into a dataframe with the exact same column
    # names as the training data or the model will complain
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

    # same encoding steps as preprocess.py, has to match exactly
    for col in BINARY_COLS:
        raw[col] = raw[col].astype("category").cat.codes

    raw = pd.get_dummies(raw, columns=CATEGORICAL_COLS, drop_first=True)
    # reindex fills in any dummy columns that didn't show up for this
    # one input (like if Mjob_teacher wasn't picked) with 0s
    raw = raw.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(raw)[0]
    # clamp it so it can't predict something weird like -2 or 25
    prediction = max(0, min(20, prediction))

    st.write(f"### predicted grade: {prediction:.1f} / 20")
    st.write(f"category: **{performance_category(prediction)}**")
