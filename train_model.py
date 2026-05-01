import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
print("Loading data...")
deliveries = pd.read_csv('deliveries.csv')
matches    = pd.read_csv('matches.csv')

print(f"Deliveries shape : {deliveries.shape}")
print(f"Matches shape    : {matches.shape}")
print(f"\nDeliveries columns: {deliveries.columns.tolist()}")

# Merge match info
match_cols = ['id', 'season', 'venue', 'date',
              'team1', 'team2']
available  = [c for c in match_cols
              if c in matches.columns]
df = deliveries.merge(
    matches[available],
    left_on='match_id',
    right_on='id',
    how='left'
)

# Identify column names dynamically
bat_col  = 'batting_team'   if 'batting_team'  in df.columns \
           else 'bat_team'
bowl_col = 'bowling_team'   if 'bowling_team'  in df.columns \
           else 'bowl_team'
over_col = 'over'           if 'over'          in df.columns \
           else 'overs'
runs_col = 'total_runs'     if 'total_runs'    in df.columns \
           else 'runs_off_bat'

print(f"\nUsing columns: bat={bat_col}, "
      f"bowl={bowl_col}, over={over_col}, "
      f"runs={runs_col}")

# Build cumulative features
df = df.sort_values(['match_id', over_col])
df['cum_runs'] = df.groupby(
    'match_id')[runs_col].cumsum()
df['cum_wickets'] = df.groupby('match_id')[
    'player_dismissed'].transform(
    lambda x: x.notna().cumsum()
) if 'player_dismissed' in df.columns else 0

# Final score per innings
final_scores = df.groupby('match_id').agg(
    total_score=(runs_col, 'sum'),
    batting_team=(bat_col, 'first'),
    bowling_team=(bowl_col, 'first'),
    venue=('venue', 'first') if 'venue' in df.columns
          else (bat_col, 'first')
).reset_index()

# Features at over 10 (halfway point)
over10 = df[df[over_col] == 10].groupby('match_id').agg(
    runs_at_10=(runs_col, 'sum'),
    wickets_at_10=('player_dismissed',
                   lambda x: x.notna().sum())
    if 'player_dismissed' in df.columns
    else (runs_col, 'count')
).reset_index()

# Merge
data = final_scores.merge(over10, on='match_id', how='inner')
data = data.dropna()
print(f"\nTraining samples: {len(data):,}")

# Encode teams
le_bat  = LabelEncoder()
le_bowl = LabelEncoder()
data['bat_enc']  = le_bat.fit_transform(
    data['batting_team'])
data['bowl_enc'] = le_bowl.fit_transform(
    data['bowling_team'])

# Features
features = ['runs_at_10', 'wickets_at_10',
            'bat_enc', 'bowl_enc']
X = data[features]
y = data['total_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
print("Training Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
print(f"\nMean Absolute Error : {mae:.2f} runs")
print(f"R² Score            : {r2:.3f}")

# Save
teams = sorted(data['batting_team'].unique().tolist())
with open('model.pkl',  'wb') as f:
    pickle.dump(model, f)
with open('le_bat.pkl', 'wb') as f:
    pickle.dump(le_bat, f)
with open('le_bowl.pkl','wb') as f:
    pickle.dump(le_bowl, f)
with open('teams.pkl',  'wb') as f:
    pickle.dump(teams, f)

print(f"\nTeams available: {teams}")
print("All files saved! Run app.py next.")