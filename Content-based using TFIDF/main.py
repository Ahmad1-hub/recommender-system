import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data.dropna(subset=['cast', 'title', 'description', 'listed_in'], inplace=True)
        self.data = self.data.reset_index(drop=True)
        self.indices = pd.Series(self.data.index, index=self.data['title'])
        self.vectorizer = TfidfVectorizer()
        self.create_combined_column()
        self.matrix = self.vectorizer.fit_transform(self.data["combined"])
        self.cosine_similarities = cosine_similarity(self.matrix, self.matrix)

    def create_combined_column(self):
        if 'combined' not in self.data.columns:
            self.data['listed_in'] = [re.sub(r'[^\w\s]', '', t) for t in self.data['listed_in']]
            self.data['cast'] = [re.sub(',', ' ', re.sub(' ', '', t)) for t in self.data['cast']]
            self.data['description'] = [re.sub(r'[^\w\s]', '', t) for t in self.data['description']]
            self.data['title'] = [re.sub(r'[^\w\s]', '', t) for t in self.data['title']]
            self.data["combined"] = self.data['listed_in'] + '  ' + self.data['cast'] + ' ' + self.data['title'] + ' ' + self.data['description']
            self.data.drop(['listed_in', 'cast', 'description'], axis=1, inplace=True)

    def recommend(self, title, num_recommendations):
        try:
            idx = self.indices[title]
        except KeyError:
            raise ValueError("Movie title not found in the dataset.")

        sim_scores = list(enumerate(self.cosine_similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        movie_indices = [i[0] for i in sim_scores]
        return self.data['title'].iloc[movie_indices]


data_path = "./dataset/netflix_titles.csv"
recommender = ContentRecommender(data_path)

title = 'Ganglands'
try:
    suggestions = recommender.recommend(title, num_recommendations=5)
    suggestions_df = pd.DataFrame(data=suggestions)
    print(suggestions_df)
except ValueError as e:
    print(str(e))
