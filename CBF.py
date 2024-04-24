import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for 60 restaurants
restaurant_names = [f"Restaurant {i}" for i in range(1, 61)]
cuisines = ["Thai", "Mexican", "American", "Asian Fusion", "Japanese", "Irish",
            "International", "Italian", "Coffee", "Mediterranean", "Middle Eastern", "Greek", "Bakery", "Pizza", "Vegan", "Seafood", "Alcohol", "Various"]
price_ranges = [1, 2, 3]  # 1: Low, 2: Medium, 3: High
average_ratings = np.round(np.random.uniform(3.5, 5.0, size=60), 1)  # Ratings between 3.5 and 5.0

restaurants_data = {
    'restaurant_id': range(1, 61),
    'name': restaurant_names,
    'cuisine': np.random.choice(cuisines, size=60),
    'price_range': np.random.choice(price_ranges, size=60),
    'average_rating': average_ratings
}

restaurants_df = pd.DataFrame(restaurants_data)

# Scaling numerical features
scaler = MinMaxScaler()
restaurants_df[['price_range', 'average_rating']] = scaler.fit_transform(restaurants_df[['price_range', 'average_rating']])

# One-hot encoding for 'cuisine'
cuisine_encoder = OneHotEncoder(sparse=True)  # Use sparse output to save memory
cuisine_encoded = cuisine_encoder.fit_transform(restaurants_df[['cuisine']])

# Create DataFrame for the encoded features
cuisine_encoded_df = pd.DataFrame(cuisine_encoded.toarray(), columns=cuisine_encoder.get_feature_names(['cuisine']))

# Combine encoded features with the rest, excluding original categorical columns
restaurants_df = pd.concat([
    restaurants_df.drop(['cuisine'], axis=1),
    cuisine_encoded_df
], axis=1)

# Example user ratings
user_ids = np.random.randint(1, 11, size=30)  # 10 users
restaurant_ids = np.random.randint(1, 61, size=30)  # 60 restaurants
ratings = np.random.randint(1, 6, size=30)  # Ratings between 1 and 5

user_ratings = {
    'user_id': user_ids,
    'restaurant_id': restaurant_ids,
    'rating': ratings
}

ratings_df = pd.DataFrame(user_ratings)

# Split data into training and testing sets
train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.2, random_state=42)


# Weight features by user ratings and aggregate features by user
train_user_data = pd.merge(train_ratings, restaurants_df, on='restaurant_id')
for feature in list(cuisine_encoded_df.columns):
    train_user_data[feature] *= train_user_data['rating']
train_user_profiles = train_user_data.groupby('user_id').sum().reindex(columns=restaurants_df.columns.drop(['restaurant_id', 'name']), fill_value=0)

# Batch cosine similarity calculation
def batch_cosine_similarity(profiles, items, batch_size=10):
    num_batches = int(np.ceil(len(profiles) / batch_size))
    results = []

    def process_batch(batch):
        start = batch * batch_size
        end = min(start + batch_size, len(profiles))
        return cosine_similarity(profiles[start:end], items)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_batch, i) for i in range(num_batches)]
        for future in futures:
            results.extend(future.result())

    return np.vstack(results)

# Convert user profiles to NumPy array for batch processing
user_profiles_array = np.array(train_user_profiles)
restaurant_features_array = np.array(restaurants_df.drop(['restaurant_id', 'name'], axis=1))
similarity_scores = batch_cosine_similarity(user_profiles_array, restaurant_features_array, batch_size=10)
similarity_df = pd.DataFrame(similarity_scores, columns=restaurants_df['restaurant_id'], index=train_user_profiles.index)
recommendations_train = similarity_df.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)


detailed_recommendations_train = recommendations_train.apply(lambda ids: restaurants_df[restaurants_df['restaurant_id'].isin(ids)][['name']].values.flatten())
# Print detailed recommendations for each user
for user_id, recs in detailed_recommendations_train.items():
    print(f"User {user_id} recommended restaurants: {recs}")

# Evaluate these recommendations against the test set
def calculate_performance_metrics(recommendations, actual_ratings):
    precision_list = []
    recall_list = []
    f1_scores = []

    for user_id in recommendations.index:
        recommended = set(recommendations.loc[user_id])
        actual = set(actual_ratings[actual_ratings['user_id'] == user_id]['restaurant_id'])

        true_positives = len(recommended & actual)
        precision = true_positives / len(recommended) if recommended else 0
        recall = true_positives / len(actual) if actual else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_scores.append(f1_score)

    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_scores)

precision, recall, f1_score = calculate_performance_metrics(recommendations_train, test_ratings)
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

# Visualization 1: Distribution of Recommended Restaurants' Ratings and Prices

# Generate the list of recommended restaurant IDs from the nested list structure
recommended_ids = [id for sublist in recommendations_train for id in sublist]  # Flatten list of lists
# Filter restaurants_df directly to include the encoded features
recommended_rests = restaurants_df[restaurants_df['restaurant_id'].isin(recommended_ids)]


plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.histplot(recommended_rests['average_rating'], bins=10, kde=True)
plt.title('Distribution of Average Ratings in Recommendations')

plt.subplot(1, 2, 2)
sns.histplot(recommended_rests['price_range'], bins=3, kde=False)
plt.title('Distribution of Price Ranges in Recommendations')
plt.show()

# Visualization 2: Similarity Heatmap between Users and Restaurants
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df, annot=False, cmap='coolwarm', cbar=True)
plt.title('User-Restaurant Similarity Heatmap')
plt.xlabel('Restaurant ID')
plt.ylabel('User ID')
plt.show()
