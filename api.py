from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from movie_recommender import MovieRecommender
import uvicorn

app = FastAPI(
    title="Movie Recommendation API",
    description="API for recommending movies based on user input with enhanced features",
    version="1.0.0"
)

# Initialize the recommender
recommender = MovieRecommender("movies_dataset_10k.csv")

class MovieRecommendationRequest(BaseModel):
    movies: List[str]
    n_recommendations: Optional[int] = 10

class MovieRecommendation(BaseModel):
    id: int
    title: str
    genres: List[str]
    overview: str
    similarity_score: float
    popularity: float
    vote_average: float
    release_date: str
    runtime: int
    genre_overlap: float
    runtime_similarity: float

class MovieRecommendationResponse(BaseModel):
    recommendations: List[MovieRecommendation]

@app.get("/")
async def root():
    return {
        "message": "Welcome to Movie Recommendation API",
        "endpoints": {
            "/recommend": "Get movie recommendations based on input movies",
            "/docs": "API documentation",
            "/redoc": "Alternative API documentation"
        }
    }

@app.post("/recommend", response_model=MovieRecommendationResponse)
async def recommend_movies(request: MovieRecommendationRequest):
    try:
        recommendations = recommender.recommend_movies(
            request.movies,
            request.n_recommendations
        )
        return {"recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 