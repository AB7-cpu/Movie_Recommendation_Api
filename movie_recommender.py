import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from typing import List, Dict
import re
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import pickle

class MovieRecommender:
    def __init__(self, data_path: str, index_dir: str = "index_data"):
        """
        Initialize the MovieRecommender with the dataset path and index directory
        """
        self.index_dir = index_dir
        
        # Check if index exists
        if not os.path.exists(os.path.join(index_dir, "movie_index.faiss")):
            raise FileNotFoundError(
                f"FAISS index not found in {index_dir}. Please run build_index.py first."
            )
            
        # Load the index and metadata
        self.index = faiss.read_index(os.path.join(index_dir, "movie_index.faiss"))
        
        with open(os.path.join(index_dir, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            self.data = metadata['data']
            self.popularity_scaler = metadata['popularity_scaler']
            self.all_genres = metadata['all_genres']
            self.tfidf = metadata['tfidf']
        
        # Load language models with error handling
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            raise ImportError("Failed to load sentence transformer. Please make sure sentence-transformers is installed correctly.")
            
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            warnings.warn(
                "spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm\n"
                "Falling back to basic text processing without NLP features."
            )
            self.nlp = None
            
    def _normalize_movie_name(self, name: str) -> str:
        """
        Normalize movie name by removing special characters and converting to lowercase
        """
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
        normalized = ' '.join(normalized.split())
        return normalized
    
    def get_movie_index(self, movie_name: str) -> int:
        """
        Get the index of a movie in the dataset with improved matching
        """
        normalized_input = self._normalize_movie_name(movie_name)
        
        exact_matches = self.data[self.data['normalized_title'] == normalized_input]
        if len(exact_matches) > 0:
            return exact_matches.index[0]
            
        partial_matches = self.data[self.data['normalized_title'].str.contains(normalized_input, case=False, na=False)]
        if len(partial_matches) > 0:
            return partial_matches.index[0]
            
        matches = self.data[self.data['title'].str.contains(movie_name, case=False, na=False)]
        if len(matches) > 0:
            return matches.index[0]
            
        raise ValueError(f"Movie '{movie_name}' not found in the dataset")
    
    def _calculate_hybrid_score(self, content_similarity: float, popularity: float, 
                              vote: float, genre_overlap: float) -> float:
        """
        Calculate hybrid score combining multiple factors
        """
        # Weights for different components
        weights = {
            'content': 0.5,  # Increased weight since we have fewer features
            'popularity': 0.2,
            'vote': 0.2,
            'genre': 0.1
        }
        
        return (
            weights['content'] * content_similarity +
            weights['popularity'] * popularity +
            weights['vote'] * vote +
            weights['genre'] * genre_overlap
        )
    
    def recommend_movies(self, input_movies: List[str], n_recommendations: int = 10) -> List[Dict]:
        """
        Recommend movies based on input movies using enhanced hybrid approach
        """
        if not 1 <= len(input_movies) <= 5:
            raise ValueError("Number of input movies must be between 1 and 5")
            
        # Get indices of input movies
        movie_indices = [self.get_movie_index(movie) for movie in input_movies]
        
        # Get vectors for input movies
        input_vectors = self.index.reconstruct_batch(movie_indices)
        
        # Calculate average vector
        avg_vector = np.mean(input_vectors, axis=0).reshape(1, -1)
        faiss.normalize_L2(avg_vector)
        
        # Search for similar movies
        k = n_recommendations + len(movie_indices)
        distances, indices = self.index.search(avg_vector, k)
        
        # Get hybrid scores
        recommendations = []
        for idx, content_similarity in zip(indices[0], distances[0]):
            if idx not in movie_indices:
                movie = self.data.iloc[idx]
                
                # Calculate genre overlap
                input_genres = set()
                for input_idx in movie_indices:
                    input_genres.update(eval(self.data.iloc[input_idx]['genres']))
                movie_genres = set(eval(movie['genres']))
                genre_overlap = len(input_genres.intersection(movie_genres)) / len(input_genres.union(movie_genres))
                
                hybrid_score = self._calculate_hybrid_score(
                    content_similarity,
                    movie['normalized_popularity'],
                    movie['normalized_vote'],
                    genre_overlap
                )
                
                overview = movie['overview'] if pd.notna(movie['overview']) else "No overview available"
                recommendations.append({
                    'id': int(movie['id']),
                    'title': movie['title'],
                    'genres': eval(movie['genres']),
                    'overview': overview,
                    'similarity_score': float(hybrid_score),
                    'popularity': float(movie['popularity']),
                    'vote_average': float(movie['vote_average']),
                    'release_date': movie['release_date'] if pd.notna(movie['release_date']) else "Unknown",
                    'genre_overlap': float(genre_overlap)
                })
                
                if len(recommendations) == n_recommendations:
                    break
                    
        # Sort by hybrid score
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return recommendations 