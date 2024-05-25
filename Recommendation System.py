import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

data = {
    'user_id': ['User1', 'User1', 'User2', 'User2', 'User3', 'User3', 'User4', 'User4', 'User5', 'User5'],
    'item_id': ['The Shawshank Redemption', 'Inception', 'The Shawshank Redemption', 'Inception', 'Inception',
                'The Godfather', 'The Shawshank Redemption', 'The Godfather', 'Inception', 'The Godfather'],
    'rating': [5, 4, 4, 3, 5, 3, 4, 5, 4, 5]
}
reader = Reader(rating_scale=(1, 5))
df = pd.DataFrame(data)
dataset = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

trainset, testset = train_test_split(dataset, test_size=0.2)

algo = KNNBasic(sim_options={'user_based': True})
algo.fit(trainset)

user_id = input("Enter your UserId(Case-sensitive): ")
n_recommendations = int(input("Enter number of recommendations you want: "))
user_items = df[df['user_id'] == user_id]['item_id'].tolist()
items_to_predict = [item for item in df['item_id'].unique() if item not in user_items]
predictions = [algo.predict(user_id, item) for item in items_to_predict]

sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

recommended_items = [prediction.iid for prediction in sorted_predictions[:n_recommendations]]
print("Recommended items for User", user_id + ":", recommended_items)