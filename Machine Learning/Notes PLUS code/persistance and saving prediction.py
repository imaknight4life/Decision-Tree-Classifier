import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns=['genre'])
# y = music_data['genre']

# model = DecisionTreeClassifier()
# model.fit(X, y)

trained_model = joblib.load('music-recommender.joblib')
# dump_model = joblib.dump(model, 'music-recommender.joblib')

predictions = model.predict([[21, 1]])
predictions

