"""Restore the R23 byte-preserving packet, with no overwrite or link traversal.

Default inputs are beside this script. --root may name an empty restore folder.
This module only interprets the packet manifest, never evidence JSON contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import uuid

SCHEMA = "byte-preserving-evidence-packet/v2"
CHUNK = 1024 * 1024
PACKET_NAME = "detailed_evidence.tar.gz"
INDEX_NAME = "detailed_evidence_index.json"
PARTS_INDEX_NAME = "detailed_evidence.parts.json"
PARTS_SCHEMA = "byte-preserving-evidence-parts/v1"
_DEVICES = {"CON", "PRN", "AUX", "NUL", *("COM" + str(i) for i in range(1, 10)),
            *("LPT" + str(i) for i in range(1, 10))}


def relative_name(value):
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise ValueError("unsafe relative evidence path: " + repr(value))
    parts = value.split("/")
    for part in parts:
        if (not part or part in {".", ".."} or part.endswith((".", " "))
                or any(ord(c) < 32 or c in ':*?<>|"' for c in part)
                or part.split(".", 1)[0].upper() in _DEVICES):
            raise ValueError("unsafe evidence path component: " + repr(part))
    return value


def no_links(path):
    """Reject symlinks, junctions/reparse points, and hard-linked regular files."""
    path = Path(os.path.abspath(path))
    for node in (*reversed(path.parents), path):
        try:
            info = node.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)):
            raise ValueError("refuse link/reparse path: " + str(node))
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            raise ValueError("refuse hard-linked file: " + str(node))
    return path


def digest_file(path):
    path = no_links(path)
    if not path.is_file():
        raise ValueError("regular file required: " + str(path))
    value = hashlib.sha256()
    count = 0
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


def destination(root, name):
    name = relative_name(name)
    root = no_links(root)
    result = no_links(root.joinpath(*name.split("/")))
    result.relative_to(root)
    # resolve() is an additional containment check, never a link workaround.
    result.resolve().relative_to(root.resolve())
    return result


def load_index(index_path):
    index_path = no_links(index_path)
    if not index_path.is_file():
        raise ValueError("regular manifest required")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("unsupported evidence manifest schema")
    if type(data.get("packet_bytes")) is not int or data["packet_bytes"] <= 0:
        raise ValueError("invalid packet byte count")
    if not isinstance(data.get("packet_sha256"), str) or not re.fullmatch("[0-9a-f]{64}", data["packet_sha256"]):
        raise ValueError("invalid packet SHA256")
    rows = data.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty or invalid evidence manifest")
    expected, folded = {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid manifest member")
        name = relative_name(row.get("relative_path"))
        if name.casefold() in folded:
            raise ValueError("duplicate/case-colliding evidence name: " + name)
        folded.add(name.casefold())
        if type(row.get("bytes")) is not int or row["bytes"] < 0:
            raise ValueError("invalid member byte count")
        if not isinstance(row.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"]):
            raise ValueError("invalid member SHA256")
        expected[name] = row
    for name in folded:
        parts = name.split("/")
        if any("/".join(parts[:i]) in folded for i in range(1, len(parts))):
            raise ValueError("file/directory conflict in evidence manifest")
    return data, expected


def load_parts_index(parts_index, packet_data):
    """Load a Git-safe split manifest bound to the original packet identity."""
    parts_index = no_links(parts_index)
    if not parts_index.is_file():
        raise ValueError("regular parts manifest required")
    data = json.loads(parts_index.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != PARTS_SCHEMA:
        raise ValueError("unsupported parts manifest schema")
    packet = data.get("packet")
    if not isinstance(packet, dict):
        raise ValueError("invalid split packet identity")
    if (packet.get("filename") != packet_data.get("packet_name")
            or packet.get("bytes") != packet_data.get("packet_bytes")
            or packet.get("sha256") != packet_data.get("packet_sha256")):
        raise ValueError("split packet identity differs from evidence index")
    limit = data.get("part_size_limit_bytes")
    if type(limit) is not int or limit <= 0 or limit > 40 * 1024 * 1024:
        raise ValueError("invalid split part size limit")
    part_size = data.get("part_size_bytes", limit)
    if type(part_size) is not int or part_size <= 0 or part_size > limit:
        raise ValueError("invalid declared split part size")
    rows = data.get("parts")
    if (not isinstance(rows, list) or not rows
            or data.get("part_count") != len(rows)):
        raise ValueError("invalid split part list")
    expected_count = (packet["bytes"] + part_size - 1) // part_size
    if len(rows) != expected_count:
        raise ValueError("split part count differs from packet size")
    result, folded, total = [], set(), 0
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.get("sequence") != number:
            raise ValueError("invalid split part sequence")
        name = row.get("filename")
        if (not isinstance(name, str) or not name or Path(name).name != name
                or "/" in name or "\\" in name or name.casefold() in folded):
            raise ValueError("unsafe or duplicate split part filename")
        folded.add(name.casefold())
        size, sha = row.get("bytes"), row.get("sha256")
        if type(size) is not int or size <= 0 or size > part_size:
            raise ValueError("invalid split part byte count")
        if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha):
            raise ValueError("invalid split part SHA256")
        if number < len(rows) and size != part_size:
            raise ValueError("non-final split part does not fill declared size")
        total += size
        result.append({"sequence": number, "filename": name, "bytes": size,
                       "sha256": sha})
    if total != packet["bytes"] or result[-1]["bytes"] > part_size:
        raise ValueError("split parts do not cover packet identity")
    return parts_index, result


def assemble_packet_from_parts(packet, parts_index, packet_data):
    """Recreate a missing packet exclusively after every part and total digest verifies."""
    packet = no_links(packet)
    if packet.exists():
        return packet, "existing"
    parts_index, rows = load_parts_index(parts_index, packet_data)
    parent = no_links(packet.parent)
    if not parent.is_dir():
        raise ValueError("packet parent directory must exist")
    temporary = parent / ("." + packet.name + ".assemble-" + uuid.uuid4().hex + ".tmp")
    value, count = hashlib.sha256(), 0
    created_packet = False
    try:
        with temporary.open("xb") as output:
            for row in rows:
                part = no_links(parts_index.parent / row["filename"])
                if digest_file(part) != (row["bytes"], row["sha256"]):
                    raise ValueError("split part bytes/SHA256 mismatch: " + row["filename"])
                with part.open("rb") as source:
                    while block := source.read(CHUNK):
                        output.write(block)
                        value.update(block)
                        count += len(block)
            output.flush()
            os.fsync(output.fileno())
        if (count, value.hexdigest()) != (packet_data["packet_bytes"], packet_data["packet_sha256"]):
            raise ValueError("assembled split parts differ from packet identity")
        try:
            os.link(temporary, packet)
        except FileExistsError:
            if digest_file(packet) != (packet_data["packet_bytes"], packet_data["packet_sha256"]):
                raise ValueError("refuse different packet created during assembly")
            return packet, "existing"
        temporary.unlink()
        created_packet = True
        if digest_file(packet) != (packet_data["packet_bytes"], packet_data["packet_sha256"]):
            raise ValueError("assembled packet verification failed")
        return packet, "assembled_from_parts"
    except Exception:
        if created_packet:
            try:
                info = packet.lstat()
                if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    packet.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def archive_members(tar, expected):
    members = tar.getmembers()
    seen = set()
    for member in members:
        relative_name(member.name)
        if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            raise ValueError("archive contains a link or non-regular member: " + member.name)
        if member.name in seen or member.name not in expected:
            raise ValueError("duplicate/unexpected archive member: " + member.name)
        seen.add(member.name)
        if member.size != expected[member.name]["bytes"]:
            raise ValueError("archive member size differs: " + member.name)
    if seen != set(expected):
        raise ValueError("archive member set differs from manifest")
    return members


def check_member(tar, member, row, *, compare_path=None):
    """Hash and optionally compare every byte, not just sizes or metadata."""
    value, count = hashlib.sha256(), 0
    other = None
    if compare_path is not None:
        compare_path = no_links(compare_path)
        if not compare_path.is_file():
            raise ValueError("evidence destination is not a regular file")
        other = compare_path.open("rb")
    try:
        with tar.extractfile(member) as source:
            while block := source.read(CHUNK):
                count += len(block)
                value.update(block)
                if other is not None and other.read(len(block)) != block:
                    raise ValueError("refuse divergent evidence: " + member.name)
        if other is not None and other.read(1):
            raise ValueError("refuse divergent evidence length: " + member.name)
    finally:
        if other is not None:
            other.close()
    if count != row["bytes"] or value.hexdigest() != row["sha256"]:
        raise ValueError("archive member SHA256 differs: " + member.name)


def restore(packet, index, root, *, verify_only=False, parts_index=None):
    packet, root = no_links(packet), no_links(root)
    data, expected = load_index(index)
    if packet.exists():
        packet_source = "existing"
    else:
        if parts_index is None:
            raise ValueError("packet is missing and no split parts manifest was provided")
        packet, packet_source = assemble_packet_from_parts(packet, parts_index, data)
    if digest_file(packet) != (data["packet_bytes"], data["packet_sha256"]):
        raise ValueError("packet bytes/SHA256 mismatch")
    if root.exists() and not root.is_dir():
        raise ValueError("restore root must be a directory")
    restored = already = 0
    with tarfile.open(packet, "r:gz") as tar:
        members = archive_members(tar, expected)
        # Validate the entire packet and ALL existing destinations before writes.
        for member in members:
            target = destination(root, member.name)
            if target.exists() and not target.is_file():
                raise ValueError("refuse existing non-file destination: " + str(target))
            check_member(tar, member, expected[member.name], compare_path=target if target.exists() else None)
            already += int(target.exists())
        if not verify_only:
            for member in members:
                target = destination(root, member.name)
                if target.exists():
                    check_member(tar, member, expected[member.name], compare_path=target)
                    continue
                no_links(target.parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                target = destination(root, member.name)
                value, count = hashlib.sha256(), 0
                with target.open("xb") as output, tar.extractfile(member) as source:
                    while block := source.read(CHUNK):
                        output.write(block)
                        value.update(block)
                        count += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                row = expected[member.name]
                if count != row["bytes"] or value.hexdigest() != row["sha256"]:
                    raise ValueError("member changed while restoring; exclusive output retained: " + member.name)
                if digest_file(target) != (row["bytes"], row["sha256"]):
                    raise ValueError("restored output verification failed: " + member.name)
                restored += 1
    if digest_file(packet) != (data["packet_bytes"], data["packet_sha256"]):
        raise ValueError("packet changed during restore")
    return {"verified_members": len(expected), "restored": restored,
            "already_verified": already, "verify_only": verify_only,
            "packet_source": packet_source}


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=here)
    parser.add_argument("--packet", type=Path, default=here / PACKET_NAME)
    parser.add_argument("--index", type=Path, default=here / INDEX_NAME)
    parser.add_argument("--parts-index", type=Path, default=here / PARTS_INDEX_NAME)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(restore(args.packet, args.index, args.root,
                             verify_only=args.verify_only,
                             parts_index=args.parts_index), ensure_ascii=False))


if __name__ == "__main__":
    main()
