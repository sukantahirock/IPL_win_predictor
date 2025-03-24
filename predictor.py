import joblib
import pandas as pd

# Load trained model
model = joblib.load("ipl_win_model_rf.pkl")

# Load encoders
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
winner_encoder = joblib.load("winner_encoder.pkl")


def predict_winner(batting_team, bowling_team, venue, first_innings_score, total_runs):
    # Check if inputs exist in trained encoders
    if batting_team not in team_encoder.classes_:
        return f"Error: Batting team '{batting_team}' not found in trained data."
    if bowling_team not in team_encoder.classes_:
        return f"Error: Bowling team '{bowling_team}' not found in trained data."
    if venue not in venue_encoder.classes_:
        return f"Error: Venue '{venue}' not found in trained data."

    # Encode inputs
    batting_team_encoded = team_encoder.transform([batting_team])[0]
    bowling_team_encoded = team_encoder.transform([bowling_team])[0]
    venue_encoded = venue_encoder.transform([venue])[0]

    # Create input dataframe
    input_data = pd.DataFrame([{
        "batting_team": batting_team_encoded,
        "bowling_team": bowling_team_encoded,
        "venue": venue_encoded,
        "first_innings_score": first_innings_score,
        "total_runs": total_runs
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    predicted_winner = winner_encoder.inverse_transform([prediction])[0]

    return predicted_winner


if __name__ == "__main__":
    winner = predict_winner("Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium", 180, 50)
    print(f"Predicted Winner: {winner}")
