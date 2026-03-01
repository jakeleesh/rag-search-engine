#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command, build_command, tf_command, idf_command, tfidf_command, bm25_idf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Cache")

    tf_parser = subparsers.add_parser("tf", help="Calculate term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    tf_parser.add_argument("term", type=str, help="Search term to find counts for")

    idf_parser = subparsers.add_parser("idf", help="Calculate Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Search term to find counts for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate term frquency - inverse document frequency")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    tfidf_parser.add_argument("term", type=str, help="Search term to find counts for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = search_command(args.query, 5)
            for i, result in enumerate(results):
                print(f"{i} {result['title']}")
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()