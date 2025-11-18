from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Requirement Extraction Pipeline")
    parser.add_argument("input", type=str, help="Input file or directory containing documents")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size for splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for splitting")
    parser.add_argument("--out", type=str, default="output.json", help="Output JSON file")
    args = parser.parse_args()

    result = asyncio.run(run_pipeline(args.input, args.chunk_size, args.chunk_overlap))
    Path(args.out).write_text(result.to_json(), encoding="utf-8")
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()


