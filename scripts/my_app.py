import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# CSS
with open('scripts/style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">MovieSet</div>', unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    df = pd.read_excel(r'docs/movies_ironhack_project.xlsx')
    return df

# Load the dataset and cache it
df = load_dataset()

# User input (movie name)
movie_input = st.text_input('PICK YOUR MOVIE')

# Random movie button
if st.button('Random Movie'):
    movie_input = random.choice(df['title'].values)

# Convert text to numeric 
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(df['combined_features'])

@st.cache_data
# Create a recommendation function
def recommend_movies(movie_title, num_recommendations=10):
    if movie_title.lower() not in df['title'].str.lower().values:
        return "not found."
    movie_index = df[df['title'].str.lower() == movie_title.lower()].index[0]
    similarity_scores = cosine_similarity(feature_vectors[movie_index], feature_vectors)
    similar_movies = list(enumerate(similarity_scores[0]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    recommended_movies = []
    for i in sorted_similar_movies:
        movie = df.iloc[i[0]]
        recommended_movies.append({
            'title': movie['title'],
            'genre': movie['genre'], 
            'director': movie['director']
        })
    
    return recommended_movies

# Execute recommendation if there is user input
if movie_input:
    similar_movies = recommend_movies(movie_input.lower())
    st.write('Movies similar to ', movie_input)
    if similar_movies != "not found.":
        # Create a box for each movie title
        html_content = '<div class="suggestions-container">'
        for movie in similar_movies:
            html_content += (
                f'<div class="suggestion-box">'
                f'  <div class="title">{movie["title"]}</div>'
                f'  <div class="details">{movie["genre"]}</div>'
                f'  <div class="details">Directed by: {movie["director"]}</div>'
                f'</div>'
            )
        html_content += '</div>'
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.write('No similar movies found.')
else:
    st.write('No movie found or selected.')
