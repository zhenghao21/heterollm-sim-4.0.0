"""Create an exclusive, byte-verified R23 evidence packet after scoring finishes.

Never deletes/moves originals or parses evidence JSON. No action occurs on import.
--list-only lists eligible paths/sizes without reading their contents or packaging.
A failed run may leave its newly created incomplete packet; it is never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import stat
import sys
import tarfile

sys.dont_write_bytecode = True
from restore_evidence import (SCHEMA, PACKET_NAME, INDEX_NAME, relative_name,
                              no_links, digest_file, check_member, archive_members)

THRESHOLD = 180 * 1024
EXCLUDED_DIRECTORIES = {"source", "sources", "source_snapshot", "source_snapshots",
                        "source-snapshot", "source-snapshots", "snapshot_source", "__pycache__", ".git"}
EXCLUDED_EXTENSIONS = {".exe", ".dll", ".obj", ".pdb", ".lib", ".pyd", ".so", ".a", ".pyc"}
RAW_EXTENSIONS = {".sqlite", ".sqlite-wal", ".sqlite-shm", ".nsys-rep", ".jsonl"}


def reason_for(path, size):
    name, suffix = path.name.casefold(), path.suffix.casefold()
    if name.endswith("_task.txt") or suffix in EXCLUDED_EXTENSIONS:
        return None
    if suffix in RAW_EXTENSIONS:
        return "raw_trace_or_jsonl"
    if suffix != ".json":
        return None
    if name == "freeze.json" or name.endswith(".freeze.json"):
        return "frozen_prediction_inputs"
    if (name.endswith(".prediction.json") or name in {"prediction.json", "predictions.json"}
            or "predictions" in (part.casefold() for part in path.parts)):
        return "prediction_result"
    return "detailed_json_above_180_kib" if size > THRESHOLD else None


def snapshot(root, excluded_names):
    root = no_links(root)
    if not root.is_dir():
        raise ValueError("evidence root must be an existing directory")
    result = []
    folded = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        kept = []
        for name in sorted(dirs):
            lower = name.casefold()
            if (lower in EXCLUDED_DIRECTORIES or lower.startswith(("source_snapshot_", "source-snapshot-"))):
                continue
            no_links(Path(base) / name)
            kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            path = Path(base) / name
            relative = path.relative_to(root)
            if relative.as_posix() in excluded_names:
                continue
            info = path.lstat()
            selected = reason_for(relative, info.st_size)
            if selected is None:
                continue
            no_links(path)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("eligible evidence is not regular: " + str(path))
            key = relative_name(relative.as_posix())
            if key.casefold() in folded:
                raise ValueError("case-colliding evidence path: " + key)
            folded.add(key.casefold())
            result.append({"relative_path": key, "bytes": info.st_size,
                           "selection_reason": selected,
                           "_signature": (info.st_size, info.st_mtime_ns, info.st_ino)})
    return sorted(result, key=lambda row: row["relative_path"])


def local_output(root, name):
    relative_name(name)
    if "/" in name:
        raise ValueError("packet/index name must be a single relative filename")
    path = no_links(root / name)
    if os.path.lexists(path):
        raise FileExistsError("exclusive-create refuses existing output: " + str(path))
    return path


def package(root, *, packet_name=PACKET_NAME, index_name=INDEX_NAME, list_only=False):
    root = no_links(root)
    if packet_name == index_name:
        raise ValueError("packet and manifest must have different names")
    for name in (packet_name, index_name):
        relative_name(name)
        if "/" in name:
            raise ValueError("output names must be single filenames")
    excluded = {packet_name, index_name, PACKET_NAME, INDEX_NAME}
    selected = snapshot(root, excluded)
    if list_only:
        return {"list_only": True, "members": [{k: v for k, v in row.items() if not k.startswith("_")}
                                               for row in selected]}
    if not selected:
        raise ValueError("no eligible evidence; refusing empty packet")
    packet, index = local_output(root, packet_name), local_output(root, index_name)
    members = []
    for row in selected:
        count, sha = digest_file(root / row["relative_path"])
        if count != row["bytes"]:
            raise ValueError("source changed after discovery: " + row["relative_path"])
        members.append({"relative_path": row["relative_path"], "bytes": count,
                        "sha256": sha, "selection_reason": row["selection_reason"]})
    if snapshot(root, excluded) != selected:
        raise ValueError("eligible source set changed before packaging")
    # Exclusive mode is essential. No existing archive, manifest or evidence is
    # replaced. Normalize only TAR metadata; member contents stay byte-exact.
    with packet.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for row in members:
                    source = no_links(root / row["relative_path"])
                    info = tarfile.TarInfo(row["relative_path"])
                    info.size, info.mode, info.mtime = row["bytes"], 0o644, 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with source.open("rb") as stream:
                        tar.addfile(info, stream)
        output.flush()
        os.fsync(output.fileno())
    expected = {row["relative_path"]: row for row in members}
    with tarfile.open(packet, "r:gz") as tar:
        for member in archive_members(tar, expected):
            check_member(tar, member, expected[member.name], compare_path=root / member.name)
    if snapshot(root, excluded) != selected:
        raise ValueError("source set changed during packaging; packet retained, no manifest written")
    count, sha = digest_file(packet)
    document = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "packet_name": packet_name, "packet_bytes": count, "packet_sha256": sha,
        "members": members, "member_count": len(members),
        "originals_preserved_locally": True, "member_bytes_compared_to_sources": True,
        "selection_policy": {"scope": "round_023 descendants only",
            "large_json_threshold_bytes": THRESHOLD, "large_json_comparison": ">",
            "all_prediction_and_freeze_json": True,
            "raw_extensions": sorted(RAW_EXTENSIONS),
            "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
            "excluded_directory_prefixes": ["source_snapshot_", "source-snapshot-"],
            "excluded_extensions": sorted(EXCLUDED_EXTENSIONS), "excluded_name_suffix": "_TASK.txt",
            "copied_source_snapshots_included": False, "evidence_payloads_parsed": False}}
    with index.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(document, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    return {"packet": str(packet), "index": str(index), "members": len(members),
            "packet_bytes": count, "packet_sha256": sha, "verified_byte_exact": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--packet-name", default=PACKET_NAME)
    parser.add_argument("--index-name", default=INDEX_NAME)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(package(args.root, packet_name=args.packet_name,
        index_name=args.index_name, list_only=args.list_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
