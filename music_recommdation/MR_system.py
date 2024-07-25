import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy

# Load your dataset
# Assume the dataset is in a CSV file with columns: user_id, song_id, rating, timestamp
from google.colab import files
uploaded = files.upload()

# Use pandas to read the CSV
df = pd.read_csv(list(uploaded.keys())[0])

# Display the first few rows of the dataframe
df.head()

# Define the format
reader = Reader(rating_scale=(1, 5))

# Load the dataset from the pandas dataframe
data = Dataset.load_from_df(df[['user_id', 'song_id', 'rating']], reader)

from google.colab import drive
drive.mount('/content/drive')

trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

# Test the algorithm on the testset
predictions = algo.test(testset)

# Compute and print Root Mean Squared Error
accuracy.rmse(predictions)

# Function to get top n recommendations for a user
def get_top_n_recommendations(predictions, n=10):
    # First map the predictions to each user.
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Get top 10 recommendations for all users
top_n_recommendations = get_top_n_recommendations(predictions, n=10)

# Display recommendations for a specific user
user_id = df['user_id'].iloc[0]
print(f"Top 10 recommendations for user {user_id}: {top_n_recommendations[user_id]}")


