# RAG Search Engine

A retrieval-augmented generation (RAG) search engine for a movie streaming service dataset, built from scratch using the Gemini API and sentence transformers.

---

## Table of Contents

- [About the Project](#about-the-project)
  - [Key Features](#key-features)
  - [Solution Architecture](#solution-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)

---

## About the Project

A search engine built from the ground up, progressing from basic keyword search through to multimodal image search and RAG. Each capability is independently explorable via its own CLI.

Built on a movies dataset. Used Gemini (via `google-genai`) because it has a free tier and `gemma-3-27b-it` is a capable model.

### Key Features

- **Keyword Search**: Custom inverted index with TF-IDF and BM25 scoring, including text preprocessing (lowercasing, punctuation removal, stop word filtering, stemming), term frequency saturation, and document length normalization.
- **Semantic Search**: Sentence embeddings (`all-MiniLM-L6-v2`) with fixed-size and semantic chunking, cosine similarity search over chunk embeddings.
- **Hybrid Search**: Combines keyword and semantic results via score normalization with weighted combination or Reciprocal Rank Fusion (RRF).
- **Query Enhancement**: LLM-powered spell correction, query rewriting, and query expansion before retrieval.
- **Re-Ranking**: Individual LLM re-ranking, batch LLM re-ranking, and cross-encoder re-ranking (`cross-encoder/ms-marco-TinyBERT-L2-v2`).
- **Retrieval-Augmented Generation**: Answer questions, summarize documents, answer with citations, and detailed question answering using retrieved context.
- **Multimodal Search**: Image-to-text search using CLIP embeddings (`clip-ViT-B-32`).
- **Evaluation**: Precision@k, Recall@k, F1 score against a golden dataset, plus LLM-as-a-judge evaluation.

### Solution Architecture

- **Data**: Movies JSON dataset with titles and descriptions
- **Keyword Layer**: Custom inverted index with BM25 scoring, persisted to disk as a pickle cache
- **Semantic Layer**: `all-MiniLM-L6-v2` sentence transformer producing chunk embeddings, stored as `.npy` files
- **Hybrid Layer**: RRF or weighted combination merging both ranked lists into a single result set
- **Query Enhancement**: Gemini LLM rewrites the query before retrieval (spell, rewrite, expand)
- **Re-Ranking**: A second-pass cross-encoder or LLM reorders the top-k candidates
- **RAG Layer**: Retrieved documents are passed as context to the Gemini LLM to generate a final answer
- **Multimodal Layer**: CLIP model encodes both images and text into a shared embedding space for image-based search
- **Evaluation**: A golden dataset drives Precision@k / Recall@k / F1 metrics and LLM-as-a-judge scoring

---

## Tech Stack

- **LLM**: Gemini API (`gemma-3-27b-it`) via `google-genai`
- **Embeddings**: `sentence-transformers` — `all-MiniLM-L6-v2` (text), `clip-ViT-B-32` (multimodal)
- **Re-Ranking**: `cross-encoder/ms-marco-TinyBERT-L2-v2`
- **Numerical**: NumPy
- **Text Processing**: NLTK (stemming, tokenization, stop words)
- **Image Processing**: Pillow
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Version Control**: Git, GitHub

---

## Installation

Create a `.env` file in the root of the project with your Gemini API key. Get the API key from [Google AI Studio](https://aistudio.google.com/).

```dotenv
GEMINI_API_KEY=<key>
```

Install dependencies:

```bash
uv sync
```

## Usage

All tools are CLI scripts inside the `cli/` directory. Run them with `uv run` from the project root.

### Keyword Search

```bash
# BM25 search
uv run cli/keyword_search_cli.py bm25search "animated adventure"

# Build the inverted index cache
uv run cli/keyword_search_cli.py build

# Inspect TF, IDF, TF-IDF, BM25 scores
uv run cli/keyword_search_cli.py tf <doc_id> <term>
uv run cli/keyword_search_cli.py idf <term>
uv run cli/keyword_search_cli.py tfidf <doc_id> <term>
uv run cli/keyword_search_cli.py bm25idf <term>
uv run cli/keyword_search_cli.py bm25tf <doc_id> <term>
```

### Semantic Search

```bash
# Search using sentence embeddings
uv run cli/semantic_search_cli.py search "space exploration"

# Search over semantic chunks
uv run cli/semantic_search_cli.py search_chunked "space exploration"

# Chunk a document (fixed-size or semantic)
uv run cli/semantic_search_cli.py chunk "some long text..." --chunk-size 200 --overlap 50
uv run cli/semantic_search_cli.py semantic_chunk "some long text..." --max-chunk-size 4
```

### Hybrid Search

```bash
# Weighted combination of BM25 + semantic (alpha controls BM25 weight)
uv run cli/hybrid_search_cli.py weighted-search "romantic comedy" --alpha 0.5

# Reciprocal Rank Fusion
uv run cli/hybrid_search_cli.py rrf-search "romantic comedy" --k 60 --limit 5

# With query enhancement
uv run cli/hybrid_search_cli.py rrf-search "romntc cmmedy" --enhance spell
uv run cli/hybrid_search_cli.py rrf-search "action movies" --enhance expand

# With re-ranking
uv run cli/hybrid_search_cli.py rrf-search "thriller" --rerank-method cross_encoder

# With LLM-as-a-judge evaluation
uv run cli/hybrid_search_cli.py rrf-search "sci-fi adventure" --evaluate
```

### Retrieval-Augmented Generation

```bash
# Answer a question using RAG
uv run cli/augmented_generation_cli.py rag "What are some good movies for kids?"

# Summarize retrieved documents
uv run cli/augmented_generation_cli.py summarize "movies about space" --limit 5

# Answer with citations
uv run cli/augmented_generation_cli.py citations "best animated films"

# Detailed question answering
uv run cli/augmented_generation_cli.py question "What makes a great heist movie?"
```

### Multimodal Search

```bash
# Search for movies using an image
uv run cli/multimodal_search_cli.py image_search path/to/image.jpg --limit 5
```

### Evaluation

```bash
# Evaluate hybrid search precision, recall, and F1 against the golden dataset
uv run cli/evaluation_cli.py evaluate --limit 5
```
