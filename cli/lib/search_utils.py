import json
# Common way is os library
from pathlib import Path


# Start from current file
# Going uo to cli
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'/'movies.json'

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    # Could return dict with just 'movies' in it, but only thing in list so return just the list, cleaner
    return data['movies']