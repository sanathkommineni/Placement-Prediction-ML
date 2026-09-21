# Placement Prediction ML

A beginner-friendly machine learning project that predicts student placement outcomes using CGPA, attendance, and CRT score.

## Features
- 300-row synthetic dataset
- Logistic Regression
- Decision Tree
- Train/test split
- Accuracy comparison
- Sample-student prediction
- Feature influence output

## Run
```bash
pip install -r requirements.txt
python src/train.py
```

## Dataset Note
The included dataset is **synthetic** and is for educational demonstration only. It is not real student or placement data and should not be used for real hiring decisions.

## Workflow
```text
Dataset → Feature Selection → Train/Test Split
       → Logistic Regression + Decision Tree
       → Accuracy Comparison → Sample Prediction
```
