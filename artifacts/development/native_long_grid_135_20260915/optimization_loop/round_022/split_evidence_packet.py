"""Split the immutable R21 evidence packet into Git-safe byte-preserving parts.

The source packet and its original member index are read-only inputs. Parts and the
parts index are created exclusively; existing files are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

CHUNK = 1024 * 1024
PART_SIZE_BYTES = 40 * 1024 * 1024
PACKET_NAME = "detailed_evidence.tar.gz"
INDEX_NAME = "detailed_evidence_index.json"
PARTS_INDEX_NAME = "detailed_evidence.parts.json"
PARTS_SCHEMA = "byte-preserving-evidence-parts/v1"


def no_links(path):
    path = Path(os.path.abspath(path))
    for node in (*reversed(path.parents), path):
        try:
            info = node.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)):
            raise ValueError("refuse link/reparse path: " + str(node))
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            raise ValueError("refuse hard-linked file: " + str(node))
    return path


def digest_file(path):
    path = no_links(path)
    if not path.is_file():
        raise ValueError("regular file required: " + str(path))
    value, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while block := stream.read(CHUNK):
            value.update(block)
            count += len(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("file changed during verification: " + str(path))
    no_links(path)
    current = path.stat()
    if (current.st_ino, current.st_size, current.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("file replaced during verification: " + str(path))
    return count, value.hexdigest()


def reference(path):
    count, sha = digest_file(path)
    return {"path": str(Path(path).resolve()), "sha256": sha, "bytes": count}


def load_packet_index(index_path):
    index_path = no_links(index_path)
    if not index_path.is_file():
        raise ValueError("regular evidence index required")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "byte-preserving-evidence-packet/v2":
        raise ValueError("unsupported evidence index schema")
    if (not isinstance(data.get("packet_name"), str) or not data["packet_name"]
            or type(data.get("packet_bytes")) is not int or data["packet_bytes"] <= 0
            or not isinstance(data.get("packet_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", data["packet_sha256"])):
        raise ValueError("invalid packet identity in evidence index")
    members = data.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("invalid evidence member list")
    return data


def publish_exclusive(path, payload):
    path = no_links(path)
    parent = no_links(path.parent)
    if not parent.is_dir():
        raise ValueError("parts output directory must exist")
    temporary = parent / ("." + path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError("refuse existing output: " + str(path)) from error
        temporary.unlink()
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def split_packet(packet, index, output_dir, parts_index, *, part_size=PART_SIZE_BYTES):
    if type(part_size) is not int or part_size <= 0 or part_size > PART_SIZE_BYTES:
        raise ValueError("part size must be an integer from 1 through 40 MiB")
    packet, index, output_dir, parts_index = (no_links(packet), no_links(index),
                                               no_links(output_dir), no_links(parts_index))
    if not output_dir.is_dir():
        raise ValueError("parts output directory must exist")
    evidence = load_packet_index(index)
    expected = (evidence["packet_bytes"], evidence["packet_sha256"])
    if digest_file(packet) != expected:
        raise ValueError("source packet differs from immutable evidence index")
    if parts_index.exists():
        raise ValueError("refuse existing parts index: " + str(parts_index))
    part_count = (expected[0] + part_size - 1) // part_size
    names = [f"{evidence['packet_name']}.part{number:03d}of{part_count:03d}"
             for number in range(1, part_count + 1)]
    paths = [no_links(output_dir / name) for name in names]
    if any(path.exists() for path in paths):
        raise ValueError("refuse existing packet part output")

    rows, total_hash, total_bytes = [], hashlib.sha256(), 0
    with packet.open("rb") as source:
        before = os.fstat(source.fileno())
        for number, (name, path) in enumerate(zip(names, paths), start=1):
            part_hash, part_bytes = hashlib.sha256(), 0
            with path.open("xb") as target:
                remaining = min(part_size, expected[0] - total_bytes)
                while remaining:
                    block = source.read(min(CHUNK, remaining))
                    if not block:
                        raise ValueError("source packet ended before expected byte count")
                    target.write(block)
                    part_hash.update(block)
                    total_hash.update(block)
                    part_bytes += len(block)
                    total_bytes += len(block)
                    remaining -= len(block)
                target.flush()
                os.fsync(target.fileno())
            if digest_file(path) != (part_bytes, part_hash.hexdigest()):
                raise ValueError("part changed while writing: " + name)
            rows.append({"sequence": number, "filename": name, "bytes": part_bytes,
                         "sha256": part_hash.hexdigest()})
        if source.read(1):
            raise ValueError("source packet exceeds immutable byte count")
        after = os.fstat(source.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("source packet changed while splitting")
    if (total_bytes, total_hash.hexdigest()) != expected:
        raise ValueError("split byte stream differs from immutable packet identity")
    if digest_file(packet) != expected:
        raise ValueError("source packet changed after splitting")

    manifest = {
        "schema": PARTS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "packet": {"filename": evidence["packet_name"], "bytes": expected[0],
                   "sha256": expected[1]},
        "part_size_limit_bytes": PART_SIZE_BYTES,
        "part_size_bytes": part_size,
        "part_count": len(rows),
        "parts": rows,
        "evidence_index_ref": reference(index),
        "member_count": len(evidence["members"]),
        "source_packet_present_at_split": True,
    }
    publish_exclusive(parts_index, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return {"packet_bytes": total_bytes, "packet_sha256": total_hash.hexdigest(),
            "part_count": len(rows), "parts_index": str(parts_index),
            "parts": rows}


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=here / PACKET_NAME)
    parser.add_argument("--index", type=Path, default=here / INDEX_NAME)
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--parts-index", type=Path, default=here / PARTS_INDEX_NAME)
    parser.add_argument("--part-size-bytes", type=int, default=PART_SIZE_BYTES)
    args = parser.parse_args(argv)
    print(json.dumps(split_packet(args.packet, args.index, args.output_dir, args.parts_index,
                                  part_size=args.part_size_bytes), ensure_ascii=False))


if __name__ == "__main__":
    main()
