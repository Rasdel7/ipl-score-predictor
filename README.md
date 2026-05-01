# IPL Match Score Predictor 🏏

Predicts IPL innings final score using mid-match
statistics at the 10-over mark.

## Live Demo
[Click here](https://ipl-score-predictor-qbnbicpfapbrzfy4kbyhkv.streamlit.app)

## Features
- Predict final score from over 10 match situation
- Visual score range with confidence interval
- Team average score analysis
- Score distribution for any team
- Feature importance visualization

## Model Details
- Algorithm: Gradient Boosting Regressor
- MAE: ~15 runs
- R²: ~0.75
- Features: Runs at 10, Wickets at 10, Teams

## Tools Used
- Python, Scikit-learn, Streamlit, Pandas, Matplotlib

## How to Run Locally
pip install streamlit scikit-learn pandas numpy matplotlib
python3 train_model.py
streamlit run app.py
