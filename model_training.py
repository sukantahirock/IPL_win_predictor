from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import load_and_process_data

# Load dataset
df = load_and_process_data()

# Feature selection
features = ["batting_team", "bowling_team", "venue", "first_innings_score", "total_runs"]
df = df[features + ["winner"]]
df.fillna(0, inplace=True)

# Encode categorical features using separate encoders
team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()
winner_encoder = LabelEncoder()

df["batting_team"] = team_encoder.fit_transform(df["batting_team"].astype(str))
df["bowling_team"] = team_encoder.transform(df["bowling_team"].astype(str))  # Same encoder for teams
df["venue"] = venue_encoder.fit_transform(df["venue"].astype(str))
df["winner"] = winner_encoder.fit_transform(df["winner"].astype(str))

# Train-test split
X = df.drop(columns=["winner"])
y = df["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train model with improved hyperparameters
model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and encoders
joblib.dump(model, "ipl_win_model_rf.pkl")
joblib.dump(team_encoder, "team_encoder.pkl")
joblib.dump(venue_encoder, "venue_encoder.pkl")
joblib.dump(winner_encoder, "winner_encoder.pkl")

print("Model and encoders saved successfully!")

# Cross-validation using StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
print(f"Stratified Cross-Validation Accuracy: {scores.mean():.2f}")
