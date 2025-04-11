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

class MovieRecommender:
    def __init__(self, data_path: str):
        """
        Initialize the MovieRecommender with the dataset path
        """
        self.data = pd.read_csv(data_path)
        self.index = None
        self.vectors = None
        self.popularity_scaler = MinMaxScaler()
        self.runtime_scaler = MinMaxScaler()
        
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
        
        self._prepare_data()
        
    def _normalize_movie_name(self, name: str) -> str:
        """
        Normalize movie name by removing special characters and converting to lowercase
        """
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
        normalized = ' '.join(normalized.split())
        return normalized
        
    def _prepare_data(self):
        """
        Prepare the data and create FAISS index with enhanced features
        """
        # Clean and combine features
        self.data['combined_features'] = self.data.apply(
            lambda x: self._create_combined_features(x),
            axis=1
        )
        
        # Create normalized movie names for better matching
        self.data['normalized_title'] = self.data['title'].apply(self._normalize_movie_name)
        
        # Normalize numerical features
        self.data['normalized_popularity'] = self.popularity_scaler.fit_transform(
            self.data[['popularity']].fillna(0)
        )
        self.data['normalized_vote'] = self.popularity_scaler.fit_transform(
            self.data[['vote_average']].fillna(0)
        )
        self.data['normalized_runtime'] = self.runtime_scaler.fit_transform(
            self.data[['runtime']].fillna(0)
        )
        
        # Create multiple types of embeddings
        self._create_embeddings()
        
    def _create_embeddings(self):
        """
        Create multiple types of embeddings and combine them
        """
        # TF-IDF embeddings for combined features
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_vectors = tfidf.fit_transform(self.data['combined_features']).toarray()
        
        # BERT embeddings for overviews
        overviews = self.data['overview'].fillna('').tolist()
        bert_vectors = self.sentence_transformer.encode(overviews)
        
        # Genre embeddings
        genre_vectors = self._create_genre_embeddings()
        
        # Runtime embeddings (normalized)
        runtime_vectors = self.data['normalized_runtime'].values.reshape(-1, 1)
        
        # Combine all embeddings
        self.vectors = np.hstack([
            tfidf_vectors,
            bert_vectors,
            genre_vectors,
            runtime_vectors
        ]).astype('float32')
        
        # Normalize the combined vectors
        faiss.normalize_L2(self.vectors)
        
        # Create and train FAISS index
        dimension = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.vectors)
        
    def _create_genre_embeddings(self) -> np.ndarray:
        """
        Create embeddings for genres using one-hot encoding
        """
        # Get all unique genres
        all_genres = set()
        for genres in self.data['genres'].dropna():
            all_genres.update(eval(genres))
        all_genres = sorted(list(all_genres))
        
        # Create one-hot encoding
        genre_vectors = np.zeros((len(self.data), len(all_genres)))
        for i, genres in enumerate(self.data['genres']):
            if pd.notna(genres):
                movie_genres = eval(genres)
                for genre in movie_genres:
                    if genre in all_genres:
                        genre_vectors[i, all_genres.index(genre)] = 1
        
        return genre_vectors
        
    def _create_combined_features(self, movie: pd.Series) -> str:
        """
        Create enhanced combined features for a movie
        """
        features = []
        
        # Add overview if available
        if pd.notna(movie['overview']):
            if self.nlp is not None:
                # Extract keywords from overview using spaCy
                doc = self.nlp(movie['overview'])
                keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
                features.extend(keywords)
            else:
                # Fallback to basic text processing
                features.append(movie['overview'])
            
        # Add genres
        if pd.notna(movie['genres']):
            genres = eval(movie['genres'])
            features.extend(genres)
            
        # Add tagline if available
        if pd.notna(movie['tagline']):
            features.append(movie['tagline'])
            
        # Add original language
        if pd.notna(movie['original_language']):
            features.append(f"language_{movie['original_language']}")
            
        # Add production countries
        if pd.notna(movie['production_countries']):
            countries = eval(movie['production_countries'])
            features.extend([f"country_{c}" for c in countries])
            
        # Add production companies
        if pd.notna(movie['production_companies']):
            companies = eval(movie['production_companies'])
            features.extend([f"company_{c}" for c in companies])
            
        # Add release year
        if pd.notna(movie['release_date']):
            try:
                year = movie['release_date'].split('-')[0]
                features.append(f"year_{year}")
            except:
                pass
                
        return ' '.join(features)
    
    def _calculate_hybrid_score(self, content_similarity: float, popularity: float, 
                              vote: float, runtime_similarity: float, 
                              genre_overlap: float) -> float:
        """
        Calculate hybrid score combining multiple factors
        """
        # Weights for different components
        weights = {
            'content': 0.4,
            'popularity': 0.15,
            'vote': 0.15,
            'runtime': 0.15,
            'genre': 0.15
        }
        
        return (
            weights['content'] * content_similarity +
            weights['popularity'] * popularity +
            weights['vote'] * vote +
            weights['runtime'] * runtime_similarity +
            weights['genre'] * genre_overlap
        )
    
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
    
    def recommend_movies(self, input_movies: List[str], n_recommendations: int = 10) -> List[Dict]:
        """
        Recommend movies based on input movies using enhanced hybrid approach
        """
        if not 1 <= len(input_movies) <= 5:
            raise ValueError("Number of input movies must be between 1 and 5")
            
        # Get indices of input movies
        movie_indices = [self.get_movie_index(movie) for movie in input_movies]
        
        # Get vectors for input movies
        input_vectors = self.vectors[movie_indices]
        
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
                
                # Calculate runtime similarity
                runtime_similarity = 1 - abs(
                    self.data.iloc[movie_indices[0]]['normalized_runtime'] - 
                    movie['normalized_runtime']
                )
                
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
                    runtime_similarity,
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
                    'runtime': int(movie['runtime']) if pd.notna(movie['runtime']) else 0,
                    'genre_overlap': float(genre_overlap),
                    'runtime_similarity': float(runtime_similarity)
                })
                
                if len(recommendations) == n_recommendations:
                    break
                    
        # Sort by hybrid score
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return recommendations

# Example usage
if __name__ == "__main__":
    recommender = MovieRecommender("movies_dataset_10k.csv")
    # Test with different variations of the same movie name
    input_movies = ["rrr", "kill"]
    recommendations = recommender.recommend_movies(input_movies)
    
    print("\nRecommendations:")
    for i, movie in enumerate(recommendations, 1):
        print(f"\n{i}. {movie['title']}")
        print(f"   Genres: {', '.join(movie['genres'])}")
        print(f"   Overview: {movie['overview'][:200] if isinstance(movie['overview'], str) else 'No overview available'}...") 

