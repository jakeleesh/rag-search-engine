from google import genai
import os
import time
import json
from dotenv import load_dotenv
from lib.search_utils import PROMPT_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

model = 'gemma-3-27b-it'
client = genai.Client(api_key=api_key)

def individual_rerank(query, documents):
    with open(PROMPT_PATH/'individual_rerank.md') as f:
        prompt = f.read()
    results = []
    for doc in documents:
        _prompt = prompt.format(
            query=query,
            title=doc['title'],
            description=doc['description']
        )
        response = client.models.generate_content(model=model, contents=_prompt)
        clean_response_text = (response.text or "").strip()
        try:
            clean_response_text = int(clean_response_text)
        except:
            print(f"failed to case {response.text} to int for {doc['title']}")
            clean_response_text = 0
        results.append(
            {
                **doc,
                'rerank_response': clean_response_text
            }
        )
        time.sleep(3)

    results = sorted(results, key=lambda x: x['rerank_response'], reverse=True)
    return results

def batch_rerank(query, documents):
    with open(PROMPT_PATH/'batch_rerank.md') as f:
        prompt = f.read()
    # Use XML tags. Move template
    _mtemp = '''<movie id="{idx}">{title}:\n{desc}\n</movie>\n'''
    doc_list_str = ''
    for idx, doc in enumerate(documents):
        doc_list_str += _mtemp.format(idx=idx, title=doc['title'], desc=doc['description'])
    _prompt = prompt.format(
        query=query,
        doc_list_str=doc_list_str
    )
    response = client.models.generate_content(model=model, contents=_prompt)
    response_parsed = json.loads(response.text.strip('```json').strip('```').strip())
    print(response_parsed)
    results = []
    for idx, doc in enumerate(documents):
        results.append(
            {
                **doc,
                # response_parsed[idx] going to just look at position in response_parsed
                # Not necessarily there. Need to find where it's at in the list
                # Actually want index location of the index of the document
                'rerank_score': response_parsed.index(idx) if idx in response_parsed else len(response_parsed)
            }
        )
        time.sleep(3)
    results = sorted(results, key = lambda x: x['rerank_score'], reverse=False)
    return results