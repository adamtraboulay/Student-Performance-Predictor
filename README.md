# Student-Performance-Predictor
A machine that accurately predicts the perfomance of a student's final grade based upon imformation such as 
time spent studying, previous grades and amount of absences.

I'm a data science student and this is one of my practice projects for learning ML basics
(pandas, sklearn, streamlit etc). Still pretty new to this so the code isn't perfect, just
messing around with the classic student performance dataset.

## Data

Using the UCI [Student Performance Data Set](https://archive.ics.uci.edu/dataset/320/student+performance)
(`data/student_data.csv`) - 395 students with stuff like demographics, family background, study habits,
and their 3 grades (G1, G2, G3).

## What's in here

- `notebooks/eda.ipynb` - poking around the data, checking distributions, some pivot tables etc
- `src/preprocess.py` - cleans up the data and turns the categorical columns into numbers
- `src/train.py` - trains a linear regression and a random forest, keeps whichever one is better
- `app.py` - little streamlit app where you can plug in a student's info and get a predicted grade
- `models/` - the saved model (just run train.py again if you want to retrain it)

## How to run it

```bash
pip install -r requirements.txt
python3 src/train.py
streamlit run app.py
```

Random forest is currently winning but the accuracy isn't amazing (R2 around 0.27), probably
because a lot of what determines a grade just isn't in this dataset. Might try tuning it more later.
