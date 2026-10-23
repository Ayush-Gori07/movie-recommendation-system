# The Movie Archive – Movie Recommendation System

A content-based movie recommendation website built on the Rotten Tomatoes dataset (16,000+ movies). Search a movie and get similar ones.

## How it works

- **Similarity:** TF-IDF on movie tags (genre, directors, cast, studio, writers, description, tomatometer status) + cosine similarity.
- **Numeric features:** runtime, ratings, counts and age are log-scaled, MinMax-scaled and added with a small weight.
- **Re-ranking:** `final_score = 0.8 * similarity + 0.2 * quality`, where quality is the mean of audience rating, critics rating and audience count.
- **Deployment:** only the top-20 neighbours of each movie are stored (`neighbors.pkl`), so the full similarity matrix is not needed.

## Tech stack

- Python, pandas, NumPy, scikit-learn, NLTK (model, in the notebook)
- FastAPI + Uvicorn (backend)
- HTML, Tailwind CSS, JavaScript (frontend, designed in Google Stitch)

## Project structure

```
├── main.py            # FastAPI backend
├── requirements.txt
├── movies.pkl         # movie details
├── neighbors.pkl      # top-20 recommendations per movie
└── static/
    ├── index.html     # frontend
    └── bg.jpg
```

## API

- `GET /search?q=&genre=&top=` – search movies by title, filter by genre
- `GET /movie/{id}` – movie details with recommended movies

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:8000`

## Dataset

Rotten Tomatoes Movies dataset (16,638 rows, 17 columns).

## Author

Created by Ayush
