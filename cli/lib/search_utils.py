import json
# Common way is os library
from pathlib import Path

BM25_K1 = 1.5

# Start from current file
# Going up to cli
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'
MOVIES_PATH = DATA_PATH/'movies.json'
STOPWORDS_PATH = DATA_PATH/'stopwords.txt'

CACHE_PATH = PROJECT_ROOT/'cache'

def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    # Could return dict with just 'movies' in it, but only thing in list so return just the list, cleaner
    return data['movies']

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read().splitlines()
    return data