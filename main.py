import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent

movies = pd.read_pickle(BASE / "movies.pkl")
top_idx, top_sc = pickle.load(open(BASE / "neighbors.pkl", "rb"))

# NaN titles (agar notebook mein drop_duplicates wali line chali ho) ko hata do
movies["movie_title"] = movies["movie_title"].fillna("")
for c in ["genre", "directors", "cast", "movie_info", "rating", "in_theaters_date"]:
    movies[c] = movies[c].fillna("")

valid = (movies["movie_title"] != "").values
title_lower = movies["movie_title"].str.lower().values
genre_lower = movies["genre"].str.lower().values
quality = movies["quality"].values
audience = movies["audience_rating"].values
critics = movies["tomatometer_rating"].values

app = FastAPI(title="Movie Recommender")


def card(i: int) -> dict:
    return {
        "id": int(i),
        "title": movies.at[i, "movie_title"],
        "genre": movies.at[i, "genre"],
        "year": movies.at[i, "in_theaters_date"][:4],
        "audience": int(audience[i]),
        "critics": int(critics[i]),
        "rating": movies.at[i, "rating"],
    }


@app.get("/search")
def search(q: str = "", genre: str = "", top: bool = False, limit: int = Query(36, le=100)):
    q = q.strip().lower()
    mask = valid.copy()
    if genre:
        mask &= np.array([genre.lower() in g for g in genre_lower])
    if top:
        mask &= (audience >= 80) & (critics >= 85)

    if q:
        starts = np.array([t.startswith(q) for t in title_lower])
        has = np.array([q in t for t in title_lower])
        mask &= has
        idx = np.where(mask)[0]
        # pehle jo q se start ho, phir quality
        order = np.lexsort((-quality[idx], ~starts[idx]))
        idx = idx[order]
    else:
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-quality[idx])]

    return {"total": int(len(idx)), "results": [card(i) for i in idx[:limit]]}


@app.get("/movie/{movie_id}")
def movie(movie_id: int):
    if movie_id < 0 or movie_id >= len(movies) or not valid[movie_id]:
        raise HTTPException(404, "Movie not found")

    m = movies.iloc[movie_id]
    detail = card(movie_id) | {
        "directors": m["directors"],
        "cast": m["cast"],
        "info": m["movie_info"],
        "runtime": int(m["runtime_in_minutes"]),
        "released": m["in_theaters_date"],
        "quality": round(float(m["quality"]) * 100),
    }

    recs, seen = [], {movies.at[movie_id, "movie_title"].lower()}
    for j in top_idx[movie_id]:
        j = int(j)
        if not valid[j]:
            continue
        t = title_lower[j]
        if t in seen:
            continue
        seen.add(t)
        recs.append(card(j))
        if len(recs) == 8:
            break

    return {"movie": detail, "recommendations": recs}


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
