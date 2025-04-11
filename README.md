# Movie Recommendation API

A FastAPI-based movie recommendation system that uses advanced embedding techniques and hybrid scoring to provide personalized movie recommendations.

## Features

- Content-based recommendation using multiple embedding techniques
- Hybrid scoring system combining content similarity, popularity, and other factors
- Support for 1-5 input movies
- Returns 10 most relevant recommendations
- Advanced text processing with spaCy and BERT embeddings

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the API

### Development Mode
```bash
python api.py
```

### Production Mode
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### GET /
Returns API information and available endpoints.

### POST /recommend
Get movie recommendations based on input movies.

Request body:
```json
{
    "movies": ["RRR", "Kill"],
    "n_recommendations": 10
}
```

Response:
```json
{
    "recommendations": [
        {
            "id": 12345,
            "title": "Movie Title",
            "genres": ["Action", "Drama"],
            "overview": "Movie description...",
            "similarity_score": 0.85,
            "popularity": 100.5,
            "vote_average": 7.8,
            "release_date": "2023-01-01",
            "runtime": 120,
            "genre_overlap": 0.75,
            "runtime_similarity": 0.9
        }
    ]
}
```

## Deployment

### Using Docker (Optional)
1. Build the image:
```bash
docker build -t movie-recommender .
```

2. Run the container:
```bash
docker run -p 8000:8000 movie-recommender
```

### Using PM2 (Recommended for production)
1. Install PM2:
```bash
npm install -g pm2
```

2. Start the API:
```bash
pm2 start "uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4" --name "movie-recommender"
```

3. Other PM2 commands:
```bash
pm2 status              # Check status
pm2 logs movie-recommender  # View logs
pm2 restart movie-recommender  # Restart the API
pm2 stop movie-recommender    # Stop the API
```

## Environment Variables

No environment variables are required for basic operation.

## Dependencies

- FastAPI
- Uvicorn
- Pandas
- NumPy
- scikit-learn
- FAISS
- sentence-transformers
- spaCy
- PyTorch

## License

[Your chosen license] 