import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


import os


dataset_path = os.path.join(os.path.dirname(__file__), 'movies.csv')
movies = pd.read_csv(dataset_path)



count = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = count.fit_transform(movies['genres'])
similarity = cosine_similarity(genre_matrix)


def recommend(title):
    try:
        idx = movies[movies['title'].str.contains(title, case=False, na=False)].index[0]
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        recommendations = [movies.iloc[i[0]]['title'] for i in sim_scores[1:6]]
        return recommendations
    except IndexError:
        return None


st.title("🎬 Movie Recommendation App")
st.write("Enter a movie name to get similar recommendations:")

movie_name = st.text_input("Movie Title")

if st.button("Recommend"):
    recommended_movies = recommend(movie_name)
    if recommended_movies:
        st.subheader("Recommended Movies:")
        for movie in recommended_movies:
            st.write(movie)
    else:
        st.error("Movie not found! Please try again.")
