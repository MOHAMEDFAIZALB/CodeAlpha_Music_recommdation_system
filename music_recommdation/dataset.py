import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set the random seed for reproducibility
np.random.seed(42)

# Generate user IDs
num_users = 1000
user_ids = np.arange(1, num_users + 1)

# Generate song IDs
num_songs = 500
song_ids = np.arange(1, num_songs + 1)

# Generate random ratings
def generate_random_ratings(num_users, num_songs, num_ratings):
    user_ids = np.random.choice(np.arange(1, num_users + 1), num_ratings)
    song_ids = np.random.choice(np.arange(1, num_songs + 1), num_ratings)
    ratings = np.random.randint(1, 6, num_ratings)  # Ratings between 1 and 5
    return user_ids, song_ids, ratings

num_ratings = 10000
user_ids, song_ids, ratings = generate_random_ratings(num_users, num_songs, num_ratings)

# Generate random timestamps within the last year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
timestamps = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(num_ratings)]

# Create the DataFrame
data = {
    'user_id': user_ids,
    'song_id': song_ids,
    'rating': ratings,
    'timestamp': timestamps
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('user_song_data.csv', index=False)

print("Sample data:")
print(df.head())
