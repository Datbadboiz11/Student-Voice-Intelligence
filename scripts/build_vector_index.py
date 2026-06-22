from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval import RetrievalConfig, SemanticSearchService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Student Voice Qdrant index.")
    parser.add_argument("--data-path", type=Path, help="CSV corpus path. Defaults to project data.")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the collection.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    service = SemanticSearchService(RetrievalConfig.from_project())
    result = service.build_index(
        data_path=args.data_path,
        recreate=args.recreate,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
