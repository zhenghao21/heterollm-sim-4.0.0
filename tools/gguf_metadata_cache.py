"""Create or verify GGUF metadata sidecars without decoding tensor weights."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.gguf_parity import read_gguf_metadata_cache, write_gguf_metadata_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("gguf", type=Path)
    build.add_argument("--output", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("gguf", type=Path)
    verify.add_argument("--cache", type=Path)
    verify.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        print(write_gguf_metadata_cache(args.gguf, args.output))
    else:
        metadata = read_gguf_metadata_cache(args.gguf, args.cache, strict=args.strict)
        print(f"ok {metadata.path} sha256={metadata.sha256}")


if __name__ == "__main__":
    main()
