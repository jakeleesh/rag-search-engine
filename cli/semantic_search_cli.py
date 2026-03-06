#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk_text, chunk_text_semantic, embed_chunks

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model loads properly")

    embed_parser = subparsers.add_parser("embed_text", help="Encode text with embedding model")
    embed_parser.add_argument("text", type=str, help="Text to be encoded")

    subparsers.add_parser("verify_embeddings", help="Verify the embedding model loads properly")

    embedquery_parser = subparsers.add_parser("embedquery", help="Encode query with embedding model")
    embedquery_parser.add_argument("query", type=str, help="User query to be encoded")

    search_parser = subparsers.add_parser("search", help="Search for a relevant movie")
    search_parser.add_argument("query", type=str, help="User query to search based on")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document")
    chunk_parser.add_argument("text", type=str, help="Document to be chunked")
    chunk_parser.add_argument("--overlap", type=int, default=200, help="Number of words in each fixed size chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words in each fixed size chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a document semantically")
    semantic_chunk_parser.add_argument("text", type=str, help="Document to be chunked")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of sentences in each fixed size chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Number of sentences in each fixed size chunk")

    subparsers.add_parser("embed_chunks", help="Create embeddings for semantic chunks")

    args = parser.parse_args()

    match args.command:
        case "embed_chunks":
            embed_chunks()
        case "semantic_chunk":
            chunk_text_semantic(args.text, args.overlap, args.max_chunk_size)
        case "chunk":
            chunk_text(args.text, args.overlap, args.chunk_size)
        case "search":
            search(args.query, args.limit)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()