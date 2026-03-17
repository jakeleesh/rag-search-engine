import json
from lib.search_utils import load_movies, PROJECT_ROOT
from lib.hybrid_search import HybridSearch

# Could put in search_utils.py - not going to reuse and load golden_dataset.json outside of evaluatio
def load_test_cases():
    with open(PROJECT_ROOT/'data'/'golden_dataset.json') as f:
        # Index into test_cases, get as a list
        test_cases = json.load(f)['test_cases']
    return test_cases

def evaluate(limit):
    print(f"k={limit}")
    test_cases = load_test_cases()
    movies = load_movies()

    hs = HybridSearch(movies)

    for test_case in test_cases:
        qry = test_case['query']
        exp = test_case['relevant_docs']

        rrf_results = hs.rrf_search(qry, k=60, limit=limit)
        relevant_cnt = 0
        relevant = []
        for rrf_result in rrf_results:
            relevant_cnt += rrf_result['title'] in exp
            if rrf_result['title'] in exp:
                relevant.append(rrf_result['title'])
        precision = relevant_cnt / limit
        retrieved = ", ".join([r['title'] for r in rrf_results])
        print(f"- Query: {qry}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {retrieved}")
        print(f"  - Relevant: {", ".join(relevant)}")