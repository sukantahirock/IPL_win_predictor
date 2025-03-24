import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("ipl_win_model_rf.pkl")

# Load encoders
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
winner_encoder = joblib.load("winner_encoder.pkl")

# Streamlit app UI
st.title("🏏 IPL Match Winner Predictor")

# Dropdown options
teams = list(team_encoder.classes_)
venues = list(venue_encoder.classes_)

# User Inputs
batting_team = st.selectbox("Select Batting Team", teams)
bowling_team = st.selectbox("Select Bowling Team", teams)
venue = st.selectbox("Select Venue", venues)
first_innings_score = st.number_input("Enter First Innings Score", min_value=50, max_value=300, step=5)
total_runs = st.number_input("Enter Total Runs in Match (Optional)", min_value=100, max_value=500, step=5)

# Prediction Function
def predict_winner(batting_team, bowling_team, venue, first_innings_score, total_runs):
    batting_team_encoded = team_encoder.transform([batting_team])[0]
    bowling_team_encoded = team_encoder.transform([bowling_team])[0]
    venue_encoded = venue_encoder.transform([venue])[0]

    input_data = pd.DataFrame([{
        "batting_team": batting_team_encoded,
        "bowling_team": bowling_team_encoded,
        "venue": venue_encoded,
        "first_innings_score": first_innings_score,
        "total_runs": total_runs
    }])

    prediction = model.predict(input_data)[0]
    predicted_winner = winner_encoder.inverse_transform([prediction])[0]

    return predicted_winner

# Prediction Button
if st.button("🏆 Predict Winner"):
    winner = predict_winner(batting_team, bowling_team, venue, first_innings_score, total_runs)
    st.success(f"🏆 Predicted Winner: {winner}")
