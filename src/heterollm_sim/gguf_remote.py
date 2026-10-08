"""Read pinned Hugging Face GGUF directories through strict HTTP ranges.

No weights are retained. Publisher LFS identities are asserted provenance,
not payload checksums computed or verified by this metadata-only reader.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

from .gguf_parity import (
    GGUFError, GGUFMetadata, _SPLIT_FILENAME, _split_coordinates,
    merge_gguf_shards, parse_gguf_directory,
)


class HTTPRangeReader:
    """Sequential reader with bounded prefix read-ahead and exact responses."""

    def __init__(self, url: str, size: int, *, opener=None, read_ahead=256 * 1024, timeout=60):
        if type(size) is not int or size < 24:
            raise GGUFError("invalid remote GGUF file size")
        if type(read_ahead) is not int or read_ahead < 1 or read_ahead > 256 * 1024:
            raise GGUFError("HTTP GGUF read-ahead must be between 1 and 256 KiB")
        self.url, self.size = url, size
        self.opener, self.timeout = opener, timeout
        # A small fixture or corrupt remote object must never trigger a
        # single full-file GET merely because it fits the prefix buffer.
        self.read_ahead = min(read_ahead, max(1, size // 4))
        self.position, self.buffer_start, self.buffer = 0, 0, b""
        self.downloaded_bytes = 0

    def tell(self):
        return self.position

    def _range(self, start, length):
        end = start + length - 1
        if length < 1 or start < 0 or end >= self.size:
            raise GGUFError("HTTP GGUF range exceeds publisher file bounds")
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}",
                                            "Accept-Encoding": "identity"})
        operation = self.opener.open if hasattr(self.opener, "open") else (self.opener or urlopen)
        try:
            with operation(request, timeout=self.timeout) as response:
                if response.status != 206:
                    raise GGUFError(f"HTTP GGUF requires 206 Partial Content; got {response.status}")
                expected = f"bytes {start}-{end}/{self.size}"
                if response.headers.get("Content-Range") != expected:
                    raise GGUFError("HTTP GGUF Content-Range disagrees with request or publisher size")
                content_length = response.headers.get("Content-Length")
                if content_length is None or content_length != str(length):
                    raise GGUFError("HTTP GGUF Content-Length disagrees with requested range")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise GGUFError("HTTP GGUF range cannot use content encoding")
                data = response.read(length + 1)
                if len(data) != length:
                    raise GGUFError("HTTP GGUF response body length disagrees with requested range")
        except GGUFError:
            raise
        except Exception as exc:
            raise GGUFError(f"HTTP GGUF range read failed at {start}-{end}: {exc}") from exc
        self.downloaded_bytes += len(data)
        return data

    def read(self, size=-1):
        if type(size) is not int or size < 0:
            raise GGUFError("HTTP GGUF reader refuses unbounded/full-payload reads")
        if size == 0:
            return b""
        if self.position + size > self.size:
            raise GGUFError("truncated remote GGUF directory exceeds publisher file bounds")
        chunks = []
        remaining = size
        while remaining:
            available = self.buffer_start + len(self.buffer) - self.position
            if available <= 0:
                length = min(self.read_ahead, self.size - self.position)
                # Validate the first 24-byte GGUF header before read-ahead.
                if self.position == 0:
                    length = min(length, 24)
                self.buffer_start = self.position
                self.buffer = self._range(self.position, length)
                available = len(self.buffer)
            take = min(remaining, available)
            start = self.position - self.buffer_start
            chunks.append(self.buffer[start:start + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)


def read_remote_gguf_metadata(source: Mapping, *, allow_unknown_types=False,
                              opener=None, read_ahead=256 * 1024, timeout=60) -> GGUFMetadata:
    """Read one complete pinned repo/revision/files GGUF group.

    ``files`` must contain exactly one unsplit file or all shards of one
    model. Each entry requires rfilename, size, and lfs.sha256 from the
    publisher. Revisions must be immutable commit hashes, never branches.
    """
    if not isinstance(source, Mapping):
        raise GGUFError("remote GGUF source must be a mapping")
    repo, revision, files = source.get("repo"), source.get("revision"), source.get("files")
    if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise GGUFError("invalid Hugging Face GGUF repo")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision):
        raise GGUFError("remote GGUF requires a fixed commit revision")
    if not isinstance(files, (list, tuple)) or not files:
        raise GGUFError("remote GGUF requires a nonempty file list")
    prepared, names = [], set()
    for item in files:
        if not isinstance(item, Mapping):
            raise GGUFError("invalid remote GGUF file entry")
        name, size, lfs = item.get("rfilename"), item.get("size"), item.get("lfs")
        if (not isinstance(name, str) or not name.lower().endswith(".gguf")
                or name.startswith("/") or "\\" in name or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise GGUFError("invalid remote GGUF filename")
        if name in names:
            raise GGUFError("duplicate remote GGUF file")
        names.add(name)
        digest = lfs.get("sha256") if isinstance(lfs, Mapping) else None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise GGUFError("remote GGUF requires publisher LFS SHA256 identity")
        if type(size) is not int or size < 24 or ("size" in lfs and lfs["size"] != size):
            raise GGUFError("remote GGUF publisher sizes disagree or are invalid")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(name, safe='/')}"
        prepared.append((url, {"path": url, "repo": repo, "revision": revision,
                               "rfilename": name, "size": size, "sha256": digest.lower(),
                               "sha256_origin": "publisher_lfs"}))
    shards = []
    for url, identity in prepared:
        reader = HTTPRangeReader(url, identity["size"], opener=opener, read_ahead=read_ahead, timeout=timeout)
        gguf = parse_gguf_directory(reader, path=url, file_size=identity["size"],
                                    digest=identity["sha256"], source=identity,
                                    allow_unknown_types=allow_unknown_types)
        match = _SPLIT_FILENAME.fullmatch(Path(identity["rfilename"]).name)
        coordinate = _split_coordinates(gguf)
        if match and coordinate != (int(match[2]) - 1, int(match[3])):
            raise GGUFError("remote GGUF split filename disagrees with split metadata")
        if coordinate is not None and coordinate[1] > 1 and not match:
            raise GGUFError("remote GGUF split metadata requires a canonical shard filename")
        shards.append(gguf)
    return merge_gguf_shards(shards)


def read_remote_gguf_sources(sources: Mapping | str | Path, **kwargs) -> dict[str, GGUFMetadata]:
    """Read a source JSON mapping keyed by preset name; failures stay explicit."""
    if isinstance(sources, (str, Path)):
        sources = json.loads(Path(sources).read_text(encoding="utf-8"))
    if not isinstance(sources, Mapping):
        raise GGUFError("remote GGUF source collection must be a mapping")
    return {key: read_remote_gguf_metadata(source, **kwargs) for key, source in sources.items()}


__all__ = ["HTTPRangeReader", "read_remote_gguf_metadata", "read_remote_gguf_sources"]
