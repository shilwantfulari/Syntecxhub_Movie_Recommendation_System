import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_and_clean_data():
    """
    Loads the dataset and performs basic cleaning and EDA.
    """
    print("--- Loading Data ---")
    # Load movies data (adjust filename if necessary)
    try:
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
    except FileNotFoundError:
        print("Error: 'movies.csv' or 'ratings.csv' not found. Please download the MovieLens dataset.")
        return None

    # Basic EDA (Exploratory Data Analysis)
    print(f"Movies shape: {movies.shape}")
    print(f"Ratings shape: {ratings.shape}")
    
    # Data Cleaning: Handle missing values (if any)
    movies['genres'] = movies['genres'].fillna('')
    
    # Preview data
    print("\nSample Movies:")
    print(movies.head())
    
    return movies

def build_recommender(movies):
    """
    Builds a content-based recommender system using TF-IDF on genres.
    """
    print("\n--- Building Recommender Model ---")
    
    # 1. Text Feature Extraction
    # We use TF-IDF Vectorizer to convert genres into a matrix of numbers
    # This fulfills the requirement for "text features"
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Construct the TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    
    # 2. Compute Cosine Similarity
    # This fulfills the requirement for "cosine similarity"
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

def get_recommendations(title, movies, cosine_sim):
    """
    Get top 10 movie recommendations based on similarity to the input title.
    """
    # Create a reverse mapping of indices and movie titles
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    
    # Check if title exists
    if title not in indices:
        return [f"Movie '{title}' not found in dataset."]
    
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies (ignoring the movie itself at index 0)
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices].tolist()

def main():
    # 1. Load Data
    movies = load_and_clean_data()
    if movies is None:
        return

    # 2. Build Model
    cosine_sim = build_recommender(movies)
    
    # 3. Qualitative Evaluation (Sample Queries)
    # This fulfills the requirement to "Evaluate qualitatively and show examples"
    test_movies = ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)']
    
    print("\n--- Generating Recommendations ---")
    
    for movie in test_movies:
        print(f"\nRecommendations for '{movie}':")
        recommendations = get_recommendations(movie, movies, cosine_sim)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

if __name__ == "__main__":
    main()