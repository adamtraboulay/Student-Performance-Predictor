# Student-Performance-Predictor

A machine that predicts a student's final grade based on information such as
time spent studying, previous grades, and amount of absences.

I'm a data science student and this is one of my practice projects for learning ML basics
(pandas, sklearn, Flask etc). Still pretty new to this so the code isn't perfect, just
messing around with the classic student performance dataset.

## Data

Using the UCI [Student Performance Data Set](https://archive.ics.uci.edu/dataset/320/student+performance)
(`data/student_data.csv`) - 395 students with variables like demographics, family background, study habits,
and their 3 grades (G1, G2, G3).

## Two models

Predicting G3 from G1/G2 (a student's earlier grades) is almost cheating - they're
extremely correlated - so this trains two separate models instead of picking one:

- **Blind model**: only demographics/habits/family background, no prior grades.
  R² ≈ 0.26-0.29 (5-fold CV). That's roughly the ceiling for this dataset without
  prior grades - there just isn't much more signal in "does the student have
  internet at home" than that.
- **Full model**: same features plus G1/G2. R² ≈ 0.83-0.89. Knowing a student's
  recent grades tells you most of what you need to know about their next one.

The web app lets you toggle between the two, so the R² contrast is part of the
point rather than hidden behind one number.

## What's in here

- `notebooks/eda.ipynb` - poking around the data, checking distributions, some pivot tables etc
- `src/preprocess.py` - cleans up the data, encodes categorical columns, builds the two feature sets
- `src/train.py` - cross-validated model selection (Ridge / Random Forest / Gradient Boosting) per feature set
- `backend/main.py` - Flask app: serves the frontend and a `/api/predict` endpoint
- `static/` - the frontend (plain HTML/CSS/JS, no framework)
- `models/` - the saved models (just run `train.py` again if you want to retrain them)
- `render.yaml` - deploy config for Render

## How to run it locally

```bash
pip install -r requirements.txt
python3 src/train.py
python3 backend/main.py
```

Then open http://localhost:5050.

## Deploying

`render.yaml` is set up for a one-click deploy on [Render](https://render.com):
connect this repo as a new Web Service and it picks up the build/start commands
automatically. The trained `.pkl` files are committed to the repo, so no retraining
happens at deploy time.
