import argparse
from lib.multimodal_search import verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding loads")
    verify_parser.add_argument("image_fpath", type=str, help="path to the image to process")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_fpath)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()