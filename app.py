import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Match Score Predictor")
st.markdown("Predict the final innings score using "
            "mid-match statistics — powered by ML.")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl',  'rb') as f:
        model = pickle.load(f)
    with open('le_bat.pkl', 'rb') as f:
        le_bat = pickle.load(f)
    with open('le_bowl.pkl','rb') as f:
        le_bowl = pickle.load(f)
    with open('teams.pkl',  'rb') as f:
        teams = pickle.load(f)
    return model, le_bat, le_bowl, teams

model, le_bat, le_bowl, teams = load_model()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Predict Score",
    "📊 Team Analysis",
    "📈 Model Info"
])

# Tab 1 — Predict
with tab1:
    st.markdown("### Enter Match Situation at Over 10")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox(
            "Batting Team:", teams)
        runs_at_10   = st.slider(
            "Runs scored at over 10:",
            0, 120, 65)
    with col2:
        bowling_team = st.selectbox(
            "Bowling Team:",
            [t for t in teams if t != batting_team]
        )
        wickets_at_10 = st.slider(
            "Wickets fallen at over 10:",
            0, 5, 2)

    if st.button("🎯 Predict Final Score",
                 type="primary"):
        if batting_team == bowling_team:
            st.error("Batting and bowling teams "
                     "cannot be the same!")
        else:
            try:
                bat_enc  = le_bat.transform(
                    [batting_team])[0]
                bowl_enc = le_bowl.transform(
                    [bowling_team])[0]
            except ValueError:
                bat_enc  = 0
                bowl_enc = 1

            features = np.array([[
                runs_at_10, wickets_at_10,
                bat_enc, bowl_enc
            ]])
            prediction = model.predict(features)[0]
            low  = max(0, prediction - 15)
            high = prediction + 15

            st.markdown("---")
            st.markdown(
                f"<h2 style='text-align:center; "
                f"color:#f39c12'>🏏 Predicted Score: "
                f"{prediction:.0f} runs</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align:center; "
                f"color:gray'>Expected range: "
                f"{low:.0f} – {high:.0f} runs</p>",
                unsafe_allow_html=True
            )

            # Score range visualization
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.barh(['Score Range'],
                    [high - low],
                    left=[low],
                    color='#f39c12',
                    alpha=0.4,
                    height=0.4)
            ax.axvline(x=prediction,
                       color='#e74c3c',
                       linewidth=3,
                       label=f'Predicted: '
                             f'{prediction:.0f}')
            ax.set_xlim(50, 250)
            ax.set_title(
                f'{batting_team} vs {bowling_team} — '
                f'Score Prediction',
                fontsize=13
            )
            ax.set_xlabel('Runs')
            ax.legend(fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)

            # Run rate info
            rr_10     = runs_at_10 / 10
            rr_needed = (prediction - runs_at_10) / 10
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Run Rate",
                      f"{rr_10:.2f}")
            c2.metric("Required RR (remaining)",
                      f"{rr_needed:.2f}")
            c3.metric("Wickets in Hand",
                      f"{10 - wickets_at_10}")

# Tab 2 — Team Analysis
with tab2:
    st.markdown("### Team Performance Analysis")

    deliveries = pd.read_csv('deliveries.csv')
    matches    = pd.read_csv('matches.csv')

    bat_col  = 'batting_team' \
               if 'batting_team' in deliveries.columns \
               else 'bat_team'
    runs_col = 'total_runs' \
               if 'total_runs' in deliveries.columns \
               else 'runs_off_bat'

    team_scores = deliveries.groupby(
        ['match_id', bat_col]
    )[runs_col].sum().reset_index()
    team_avg    = team_scores.groupby(
        bat_col)[runs_col].mean().sort_values(
        ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = plt.cm.RdYlGn(
        np.linspace(0.3, 1.0, len(team_avg)))[::-1]
    bars    = ax.bar(team_avg.index,
                     team_avg.values,
                     color=colors,
                     edgecolor='black')
    for bar, val in zip(bars, team_avg.values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.0f}',
            ha='center', fontsize=8,
            fontweight='bold'
        )
    ax.set_title('Average Innings Score by Team',
                 fontsize=14)
    ax.set_ylabel('Average Score')
    ax.set_xlabel('Team')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Score distribution
    st.markdown("### Score Distribution")
    selected_team = st.selectbox(
        "Select team:", sorted(teams))
    team_data = team_scores[
        team_scores[bat_col] == selected_team
    ][runs_col]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.hist(team_data, bins=30,
             color='#3498db', edgecolor='black',
             alpha=0.8)
    ax2.axvline(team_data.mean(),
                color='#e74c3c',
                linewidth=2,
                label=f'Mean: {team_data.mean():.0f}')
    ax2.axvline(team_data.median(),
                color='#2ecc71',
                linewidth=2,
                linestyle='--',
                label=f'Median: {team_data.median():.0f}')
    ax2.set_title(
        f'{selected_team} — Score Distribution',
        fontsize=13)
    ax2.set_xlabel('Innings Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

# Tab 3 — Model Info
with tab3:
    st.markdown("### Model Details")
    stats = pd.DataFrame({
        'Metric': [
            'Algorithm',
            'Training Data',
            'Features Used',
            'Mean Absolute Error',
            'R² Score',
            'Prediction Window'
        ],
        'Value': [
            'Gradient Boosting Regressor',
            'IPL 2008–2020 deliveries',
            'Runs at 10, Wickets at 10, '
            'Batting team, Bowling team',
            '~15 runs',
            '~0.75',
            'After 10 overs (halfway)'
        ]
    })
    st.dataframe(stats, use_container_width=True,
                 hide_index=True)

    st.markdown("### Feature Importance")
    feat_names = ['Runs at 10', 'Wickets at 10',
                  'Batting Team', 'Bowling Team']
    importances = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ['#3498db', '#e74c3c',
               '#2ecc71', '#f39c12']
    bars    = ax.bar(feat_names, importances,
                     color=colors, edgecolor='black')
    for bar, val in zip(bars, importances):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{val:.3f}',
            ha='center', fontsize=11,
            fontweight='bold'
        )
    ax.set_title('What Predicts the Score Most?',
                 fontsize=13)
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "IPL Score Predictor | "
    "Data: IPL 2008–2020"
)