import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np

class RecommenderSystemMetaDeta:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data.dropna(subset = ['title','type', 'director', 'cast', 'rating'], inplace = True)
        self.data = self.data.reset_index(drop=True)
        self.data = self.data[['title','type', 'director', 'cast', 'rating']]
        self.indices = pd.Series(self.data.index, index=self.data['title'])

        self.encoded_data = self.word_label(self)

    @staticmethod
    def word_label(self):
        type_encode = pd.DataFrame(LabelEncoder.fit_transform(self.data['type'],self.data['type']), columns = ['type'])
        director_encode = pd.DataFrame(LabelEncoder.fit_transform(self.data['director'], self.data['director']), columns = ['director'])
        cast_encode = pd.DataFrame(LabelEncoder.fit_transform(self.data['cast'], self.data['cast']), columns = ['cast'])
        rating_encode = pd.DataFrame(LabelEncoder.fit_transform(self.data['rating'], self.data['rating']), columns = ['rating'])

        encoded_data = pd.concat([self.data['title'], type_encode, director_encode, cast_encode, rating_encode], axis=1)
        return encoded_data

    def recommended_movies(self, data, num_recommendations):
        try:
            title_match = self.encoded_data.loc[self.encoded_data['title'] == data]
        except KeyError:
            raise ValueError("Movie title not found in the dataset")

        single_row_features = title_match.drop('title', axis=1)
        similarity_scores = cosine_similarity(single_row_features, self.encoded_data.drop('title', axis=1))
        nearest_indices = similarity_scores.argsort()[0][::-1]
        nearest_matches = self.encoded_data.loc[nearest_indices[1:num_recommendations]]
        print(nearest_matches['title'].reset_index(drop=True))

data_path = 'netflix_titles.csv'

rs = RecommenderSystemMetaDeta(data_path)
rs.recommended_movies("Ganglands", num_recommendations=6)
