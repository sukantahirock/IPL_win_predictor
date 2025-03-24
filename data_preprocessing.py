import pandas as pd


def load_and_process_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    # Rename 'id' column in matches to 'match_id' so it matches with deliveries
    matches.rename(columns={'id': 'match_id'}, inplace=True)

    # Merge datasets
    merged_df = deliveries.merge(matches, on="match_id", how="left")

    # First innings total
    first_innings_score = merged_df[merged_df["inning"] == 1].groupby("match_id")["total_runs"].sum().reset_index()
    first_innings_score.columns = ["match_id", "first_innings_score"]
    merged_df = merged_df.merge(first_innings_score, on="match_id", how="left")

    return merged_df


if __name__ == "__main__":
    df = load_and_process_data()
    print(df.head())
