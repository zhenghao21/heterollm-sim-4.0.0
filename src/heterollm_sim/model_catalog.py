"""Pinned bundled catalog plus explicit, safe Hugging Face imports.

Network access occurs only when :meth:`ModelCatalog.import_repository` or
:meth:`HuggingFaceClient.remote_search` is called.  Import reads repository
metadata and ``config.json`` only; it never downloads weights or executes
``auto_map``/remote Python code.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .ir import MTPBranchSpec
from .model_presets import (
    APPROXIMATION,
    CATALOG_CUTOFF_AT,
    CATALOG_VERSION,
    EXACT,
    OUT_OF_DOMAIN,
    ArchitectureEvidence,
    LayerPattern,
    LinearAttentionPattern,
    PresetDefinition,
    list_model_presets,
    model_preset_definition_detail,
    model_preset_metadata,
)


HF_ORIGIN = "https://huggingface.co"
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_API_BYTES = 2 * 1024 * 1024
MAX_CACHE_RECORD_BYTES = 4 * 1024 * 1024
MAX_REMOTE_SEARCH_LIMIT = 100

_REPO_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_DENSE_MODEL_TYPES = frozenset(
    {
        "bloom",
        "gpt_neox",
        "granite",
        "llama",
        "olmo",
        "olmo2",
        "opt",
        "qwen2",
        "qwen3",
        "starcoder2",
    }
)
_OSI_LICENSES = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "isc",
        "mpl-2.0",
    }
)
_NONCOMMERCIAL_LICENSE_MARKERS = ("non-commercial", "noncommercial", "-nc", "research")
_IMPORT_DTYPE_ALIASES: Mapping[str, str] = {
    "float32": "fp32",
    "fp32": "fp32",
    "f32": "fp32",
    "float16": "fp16",
    "fp16": "fp16",
    "f16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float8": "fp8",
    "fp8": "fp8",
    "float8e4m3fn": "fp8",
    "float8e5m2": "fp8",
    "fp8e4m3fn": "fp8",
    "fp8e5m2": "fp8",
    "int8": "int8",
    "uint8": "uint8",
    "int4": "int4",
    "uint4": "uint4",
}
_LICENSE_POLICY_ALIASES: Mapping[str, Tuple[str, str]] = {
    "mit license": ("open_source", "allowed"),
    "modified mit": ("open_weight", "conditional"),
    "modified-mit": ("open_weight", "conditional"),
    "modified mit license": ("open_weight", "conditional"),
    "kimi-k3": ("open_weight", "conditional"),
    "kimi k3": ("open_weight", "conditional"),
    "kimi k3 license": ("open_weight", "conditional"),
}


class CatalogError(ValueError):
    """Structured validation/import failure safe to expose through the API."""

    def __init__(self, code: str, message: str, *, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class RemoteCatalogError(CatalogError):
    """Remote endpoint, response-size, or JSON failure."""


def _is_allowlisted_hf_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError):
        return False
    return (
        parsed.scheme == "https"
        and (parsed.hostname or "").lower() == "huggingface.co"
        and port in {None, 443}
    )


class _HuggingFaceRedirectHandler(HTTPRedirectHandler):
    """Reject a redirect before urllib can forward credentials off-origin."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, Any],
        newurl: str,
    ) -> Optional[Request]:
        if not _is_allowlisted_hf_url(newurl):
            raise RemoteCatalogError(
                "remote_redirect_forbidden",
                "remote redirect left the allowlisted Hugging Face origin",
                status=502,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_repo_id(repo_id: Any) -> str:
    if not isinstance(repo_id, str) or not _REPO_ID.fullmatch(repo_id):
        raise CatalogError(
            "invalid_repo_id",
            "repo_id must be a canonical owner/name Hugging Face repository ID",
            status=400,
        )
    if any(part in {".", ".."} for part in repo_id.split("/")):
        raise CatalogError("invalid_repo_id", "repo_id contains an invalid path segment", status=400)
    return repo_id


def validate_revision(revision: Any) -> str:
    if revision is None:
        return "main"
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise CatalogError(
            "invalid_revision",
            "revision must contain only letters, digits, '.', '_', '-', and '/'",
            status=400,
        )
    if revision.startswith("/") or ".." in revision.split("/"):
        raise CatalogError("invalid_revision", "revision contains an invalid path segment", status=400)
    return revision


def default_catalog_cache_dir() -> Path:
    """Return the per-user cache path without creating it."""

    configured = os.environ.get("HETEROLLM_SIM_CATALOG_CACHE")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root) / "heterollm-sim" / "model-catalog"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "heterollm-sim" / "model-catalog"
    return Path.home() / ".cache" / "heterollm-sim" / "model-catalog"


class HuggingFaceClient:
    """Small allowlisted JSON client for explicit Hugging Face operations."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Optional[Callable[..., Any]] = None,
        token: Optional[str] = None,
    ) -> None:
        if timeout <= 0 or timeout > 60:
            raise ValueError("timeout must be in (0, 60]")
        self.timeout = float(timeout)
        self._opener = opener or build_opener(
            _HuggingFaceRedirectHandler()
        ).open
        self._token = token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

    def remote_search(self, query: Any, limit: Any = 20) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise CatalogError("invalid_query", "query must be a non-empty string", status=400)
        query = query.strip()
        if len(query) > 200:
            raise CatalogError("invalid_query", "query must be at most 200 characters", status=400)
        if isinstance(limit, bool):
            raise CatalogError("invalid_limit", "limit must be an integer", status=400)
        try:
            parsed_limit = int(limit)
        except (TypeError, ValueError, OverflowError):
            raise CatalogError("invalid_limit", "limit must be an integer", status=400)
        if not 1 <= parsed_limit <= MAX_REMOTE_SEARCH_LIMIT:
            raise CatalogError(
                "invalid_limit",
                "limit must be between 1 and {}".format(MAX_REMOTE_SEARCH_LIMIT),
                status=400,
            )
        params = urlencode(
            {
                "search": query,
                "pipeline_tag": "text-generation",
                "limit": parsed_limit,
                "full": "true",
                "sort": "downloads",
                "direction": "-1",
            }
        )
        payload = self._get_json("/api/models?{}".format(params), MAX_API_BYTES)
        if not isinstance(payload, list):
            raise RemoteCatalogError("bad_remote_json", "Hugging Face search response must be an array", status=502)
        candidates: List[Dict[str, Any]] = []
        for item in payload[:parsed_limit]:
            if not isinstance(item, dict) or item.get("private") is True:
                continue
            repo_id = item.get("id") or item.get("modelId")
            try:
                repo_id = validate_repo_id(repo_id)
            except CatalogError:
                continue
            card = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
            license_id = card.get("license") or _license_from_tags(item.get("tags"))
            gated = item.get("gated")
            candidates.append(
                {
                    "repo_id": repo_id,
                    "sha": item.get("sha") if isinstance(item.get("sha"), str) else None,
                    "author": item.get("author") or repo_id.split("/", 1)[0],
                    "last_modified": item.get("lastModified"),
                    "tags": [tag for tag in item.get("tags", []) if isinstance(tag, str)][:40],
                    "license": _license_text(license_id),
                    "access": "gated" if gated not in {False, None, "false"} else "public",
                }
            )
        return candidates

    def repository_config(
        self, repo_id: Any, revision: Any = "main"
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
        repo_id = validate_repo_id(repo_id)
        revision = validate_revision(revision)
        owner, name = repo_id.split("/", 1)
        info_path = "/api/models/{}/{}/revision/{}".format(
            quote(owner, safe=""), quote(name, safe=""), quote(revision, safe="")
        )
        info = self._get_json(info_path, MAX_API_BYTES)
        if not isinstance(info, dict):
            raise RemoteCatalogError("bad_remote_json", "repository metadata must be an object", status=502)
        if info.get("private") is True:
            raise CatalogError("private_repository", "private repositories are not importable", status=403)
        resolved_sha = info.get("sha")
        if not isinstance(resolved_sha, str) or not _COMMIT.fullmatch(resolved_sha):
            raise RemoteCatalogError("missing_resolved_sha", "repository metadata did not provide a resolved commit", status=502)
        config_path = "/{}/resolve/{}/config.json".format(
            "/".join(quote(part, safe="") for part in repo_id.split("/")),
            resolved_sha,
        )
        raw = self._get_bytes(config_path, MAX_CONFIG_BYTES)
        config_hash = hashlib.sha256(raw).hexdigest()
        try:
            config = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteCatalogError("bad_config_json", "config.json is not valid UTF-8 JSON", status=502) from exc
        if not isinstance(config, dict):
            raise RemoteCatalogError("bad_config_json", "config.json must contain a JSON object", status=502)
        return info, config, resolved_sha, config_hash

    def _get_json(self, path: str, max_bytes: int) -> Any:
        raw = self._get_bytes(path, max_bytes)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteCatalogError("bad_remote_json", "remote response is not valid UTF-8 JSON", status=502) from exc

    def _get_bytes(self, path: str, max_bytes: int) -> bytes:
        if not path.startswith("/"):
            raise ValueError("allowlisted Hugging Face paths must be absolute")
        url = HF_ORIGIN + path
        headers = {
            "Accept": "application/json",
            "User-Agent": "heterollm-sim/{}/catalog".format(CATALOG_VERSION),
        }
        if self._token:
            headers["Authorization"] = "Bearer {}".format(self._token)
        request = Request(url, headers=headers, method="GET")
        try:
            response = self._opener(request, timeout=self.timeout)
            with response:
                final_url = response.geturl() if hasattr(response, "geturl") else url
                if not _is_allowlisted_hf_url(final_url):
                    raise RemoteCatalogError("remote_redirect_forbidden", "remote response left the allowlisted Hugging Face origin", status=502)
                raw_length = response.headers.get("Content-Length") if hasattr(response, "headers") else None
                if raw_length:
                    try:
                        if int(raw_length) > max_bytes:
                            raise RemoteCatalogError("remote_response_too_large", "remote response exceeds the configured size limit", status=502)
                    except ValueError:
                        pass
                chunks = []
                total = 0
                while True:
                    chunk = response.read(min(65536, max_bytes + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise RemoteCatalogError("remote_response_too_large", "remote response exceeds the configured size limit", status=502)
                    chunks.append(chunk)
                return b"".join(chunks)
        except HTTPError as exc:
            if exc.code == 404:
                raise RemoteCatalogError(
                    "remote_not_found",
                    "Hugging Face returned HTTP 404",
                    status=404,
                ) from exc
            raise RemoteCatalogError(
                "remote_http_error",
                "Hugging Face returned HTTP {}".format(exc.code),
                status=502,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RemoteCatalogError("remote_unavailable", "Hugging Face request failed: {}".format(exc), status=502) from exc


class ModelCatalog:
    """Merged read view over bundled definitions and explicit cached imports."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        *,
        hf_client: Optional[HuggingFaceClient] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_catalog_cache_dir()
        self.hf_client = hf_client or HuggingFaceClient()
        self._bundled = tuple(_bundled_definitions())
        self._definitions: Dict[str, PresetDefinition] = {}
        self._runtime_variants: Dict[str, List[str]] = {}
        self._reload()

    def list_metadata(self) -> Tuple[Dict[str, Any], ...]:
        items = []
        for definition in sorted(self._definitions.values(), key=lambda item: item.preset_id):
            metadata = model_preset_metadata(definition)
            variants = list(metadata["variants"])
            for repo_id in self._runtime_variants.get(definition.preset_id, []):
                if repo_id != definition.source_repo and repo_id not in variants:
                    variants.append(repo_id)
            metadata["variants"] = sorted(variants, key=str.lower)
            items.append(metadata)
        return tuple(items)

    def detail(self, preset_id: str) -> Dict[str, Any]:
        definition = self._definitions[preset_id]
        detail = model_preset_definition_detail(definition)
        runtime = self._runtime_variants.get(preset_id, [])
        if runtime:
            variants = set(detail["preset"]["variants"])
            variants.update(item for item in runtime if item != definition.source_repo)
            detail["preset"]["variants"] = sorted(variants, key=str.lower)
        return detail

    def page(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        query: Optional[str] = None,
        family: Optional[str] = None,
        architecture: Optional[str] = None,
        model_kind: Optional[str] = None,
        support_level: Optional[str] = None,
        openness: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        if offset < 0:
            raise CatalogError("invalid_offset", "offset 必须大于或等于 0", status=400)
        if not 1 <= limit <= 200:
            raise CatalogError("invalid_limit", "limit 必须在 1 到 200 之间", status=400)
        all_items: List[Dict[str, Any]] = list(self.list_metadata())
        exact_filters = {
            "family": family,
            "architecture": architecture,
            "model_kind": model_kind,
            "support_level": support_level,
            "openness": openness,
            "source": source,
        }

        def apply_filters(source_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            items = source_items
            for field, expected in exact_filters.items():
                if expected is None:
                    continue
                normalized = expected.strip().lower()
                items = [
                    item
                    for item in items
                    if str(item.get(field, "")).lower() == normalized
                ]
            if query is not None and query.strip():
                needle = query.strip().lower()
                items = [
                    item
                    for item in items
                    if needle
                    in " ".join(
                        str(item.get(field, "")).lower()
                        for field in (
                            "id",
                            "name",
                            "family",
                            "parameter_scale",
                            "source_repo",
                        )
                    )
                ]
            return items

        items = apply_filters(all_items)
        filtered = items
        selected = filtered[offset : offset + limit]
        next_offset = offset + limit if offset + limit < len(filtered) else None
        facet_fields = ("family", "model_kind", "architecture", "support_level", "openness", "source")
        facets = {}
        for field in facet_fields:
            counts: Dict[str, int] = {}
            for item in all_items:
                value = str(item.get(field, "")).strip()
                if value:
                    counts[value] = counts.get(value, 0) + 1
            facets[field] = [
                {"value": value, "count": count}
                for value, count in sorted(counts.items(), key=lambda pair: (pair[0].lower(), pair[0]))
            ]
        return {
            "items": selected,
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset,
            "facets": facets,
            "catalog_version": CATALOG_VERSION,
            "cutoff_at": CATALOG_CUTOFF_AT,
        }

    def remote_search(self, query: Any, limit: Any = 20) -> List[Dict[str, Any]]:
        return self.hf_client.remote_search(query, limit)

    def import_repository(self, repo_id: Any, revision: Any = "main") -> Dict[str, Any]:
        repo_id = validate_repo_id(repo_id)
        revision = validate_revision(revision)
        info, config, resolved_sha, config_hash = self.hf_client.repository_config(repo_id, revision)
        definition = definition_from_huggingface(
            repo_id,
            revision,
            resolved_sha,
            config_hash,
            info,
            config,
        )
        alias_of = None
        for current in self._definitions.values():
            if (
                current.config_hash
                and current.config_hash == config_hash
                and (
                    current.source == "bundled"
                    or current.source_repo.lower() != repo_id.lower()
                )
            ):
                alias_of = current.preset_id
                break
        record = {
            "cache_version": 1,
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_sha": resolved_sha,
            "config_hash": config_hash,
            "alias_of": alias_of,
            "definition": _definition_to_json(definition),
            "repository": _repository_provenance(info),
            "config": config,
        }
        self._write_record(repo_id, record)
        self._reload()
        selected_id = alias_of or definition.preset_id
        detail = self.detail(selected_id)
        detail["provenance"] = {
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_sha": resolved_sha,
            "config_hash": config_hash,
            "source": "huggingface_import",
            "cached": True,
            "deduplicated_as": alias_of,
        }
        return detail

    def _record_path(self, repo_id: str) -> Path:
        digest = hashlib.sha256(repo_id.lower().encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / "{}.json".format(digest)

    def _write_record(self, repo_id: str, record: Mapping[str, Any]) -> None:
        encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        if len(encoded) > MAX_CACHE_RECORD_BYTES:
            raise CatalogError("cache_record_too_large", "imported catalog record is too large")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".catalog-",
                suffix=".tmp",
                dir=str(self.cache_dir),
                delete=False,
            ) as handle:
                temp_name = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, str(self._record_path(repo_id)))
        finally:
            if temp_name and os.path.exists(temp_name):
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass

    def _reload(self) -> None:
        definitions = {item.preset_id: item for item in self._bundled}
        aliases: List[Tuple[str, str]] = []
        if self.cache_dir.is_dir():
            for path in sorted(self.cache_dir.glob("*.json"), key=lambda item: item.name):
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    if path.stat().st_size > MAX_CACHE_RECORD_BYTES:
                        continue
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    definition = _definition_from_json(raw["definition"])
                    alias_of = raw.get("alias_of")
                    if isinstance(alias_of, str):
                        aliases.append((alias_of, str(raw.get("repo_id", definition.source_repo))))
                    elif definition.preset_id not in definitions:
                        definitions[definition.preset_id] = definition
                except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
        runtime_variants: Dict[str, List[str]] = {}
        for target, repo_id in aliases:
            if target in definitions:
                runtime_variants.setdefault(target, []).append(repo_id)
        self._definitions = definitions
        self._runtime_variants = runtime_variants


def definition_from_huggingface(
    repo_id: str,
    revision: str,
    resolved_sha: str,
    config_hash: str,
    info: Mapping[str, Any],
    config: Mapping[str, Any],
) -> PresetDefinition:
    """Convert config-only public facts; unknown architectures remain OOD."""

    text_config_raw = config.get("text_config")
    has_vision = isinstance(config.get("vision_config"), dict)
    text = text_config_raw if isinstance(text_config_raw, dict) else config
    model_type = str(text.get("model_type") or config.get("model_type") or "unknown").lower()
    layers = _positive_int(_first(text, "num_hidden_layers", "n_layer", "num_layers"))
    hidden = _positive_int(_first(text, "hidden_size", "n_embd", "d_model"))
    intermediate = _positive_int(_first(text, "intermediate_size", "ffn_dim", "n_inner"))
    heads = _positive_int(_first(text, "num_attention_heads", "n_head"))
    kv_heads = _positive_int(text.get("num_key_value_heads")) or heads
    vocabulary = _positive_int(_first(text, "vocab_size", "padded_vocab_size")) or 0
    max_sequence = _positive_int(_first(text, "max_position_embeddings", "seq_length", "n_positions")) or 0
    expert_count = _positive_int(_first(text, "num_experts", "num_local_experts")) or 1
    top_k = _positive_int(_first(text, "num_experts_per_tok", "num_experts_per_token")) or 1
    moe_intermediate = _positive_int(text.get("moe_intermediate_size"))
    if moe_intermediate:
        intermediate = moe_intermediate
    dtype, quantization, precision_supported, precision_limitations = (
        _import_precision(text, config)
    )

    patterns: Tuple[LayerPattern, ...] = ()
    support = OUT_OF_DOMAIN
    limitations: List[str] = list(precision_limitations)
    architecture = "{}_decoder".format(model_type.replace("-", "_"))
    model_kind = "moe" if expert_count > 1 else "dense"
    architecture_supported = False
    if model_type in {"qwen3_5_text", "qwen3_5_moe_text"}:
        patterns = _import_qwen35_patterns(
            text,
            layers,
            hidden,
            intermediate,
            heads,
            kv_heads,
            expert_count,
            top_k,
            dtype=dtype,
            quantization=quantization,
        )
        if patterns:
            architecture_supported = True
            support = APPROXIMATION if precision_supported else OUT_OF_DOMAIN
            architecture = "qwen3_5_hybrid_transformer"
            limitations.append(
                "The pinned config order and hybrid mixer geometry are retained; declared weight bytes are analytical estimates."
            )
    elif (
        model_type in _SAFE_DENSE_MODEL_TYPES
        and layers
        and hidden
        and intermediate
        and heads
        and expert_count == 1
        and _only_full_attention(text.get("layer_types"))
        and not text.get("sliding_window")
    ):
        hidden_act = str(text.get("hidden_act") or "").lower()
        # Config-only imports must preserve the FFN topology.  GELU/ReLU
        # families have two projections; treating them as SwiGLU invents a
        # gate projection and biases both weight and token-work estimates.
        gated_mlp = model_type not in {"bloom", "gpt_neox", "opt", "mpt", "falcon"}
        if hidden_act in {"gelu", "gelu_new", "gelu_pytorch_tanh", "relu", "relu2"}:
            gated_mlp = False
        attention_head_dim = _positive_int(text.get("head_dim"))
        patterns = (
            LayerPattern(
                repeat=layers,
                kind="dense",
                hidden_size=hidden,
                intermediate_size=intermediate,
                attention_heads=heads,
                kv_heads=kv_heads,
                attention_head_dim=attention_head_dim,
                dtype=dtype,
                quantization=quantization,
                gated_mlp=gated_mlp,
            ),
        )
        architecture_supported = True
        support = EXACT if precision_supported else OUT_OF_DOMAIN
        if attention_head_dim and attention_head_dim != hidden // heads:
            if precision_supported:
                support = APPROXIMATION
            limitations.append(
                "The public attention head_dim is retained in metadata and weight estimates but is not an independent LayerSpec field."
            )
    if not architecture_supported:
        limitations.append(
            "The imported architecture is not in the config-only materializer allowlist; metadata is retained fail-closed."
        )
    if config.get("auto_map") or text.get("auto_map"):
        limitations.append("Remote-code auto_map declarations were ignored and never executed.")

    license_id = _license_from_info(info)
    openness, commercial_use = _license_policy(license_id)
    access = "gated" if info.get("gated") not in {False, None, "false"} else "public"
    modalities = ("text", "image", "video") if has_vision else ("text",)
    unsupported = ("vision_encoder", "multimodal_projector") if has_vision else ()
    coverage = "text_backbone_only" if has_vision else ("full_language_model" if support != OUT_OF_DOMAIN else "metadata_only")
    if has_vision:
        limitations.append("Vision and multimodal projector subgraphs are excluded from ModelSpec.")
    if support == OUT_OF_DOMAIN:
        coverage = "metadata_only"

    name = repo_id.split("/", 1)[1]
    family = _family_from_name(name)
    return PresetDefinition(
        preset_id=_import_preset_id(repo_id),
        name=name,
        family=family,
        parameter_scale=_scale_from_name(name),
        source_repo=repo_id,
        source_revision=revision,
        source_sha=resolved_sha,
        config_hash=config_hash,
        source="huggingface_import",
        license=license_id,
        openness=openness,
        access=access,
        commercial_use=commercial_use,
        support_level=support,
        notes="Explicit config-only Hugging Face import; no weights or remote code were loaded.",
        vocabulary_size=vocabulary,
        max_sequence_length=max_sequence,
        patterns=patterns,
        architecture=architecture,
        modalities=modalities,
        supported_modalities=("text",),
        unsupported_subgraphs=unsupported,
        limitations=tuple(limitations),
        coverage=coverage,
        model_kind_override=model_kind,
        layer_count_override=layers or 0,
        mtp=MTPBranchSpec(
            prediction_layers=(
                _positive_int(text.get("mtp_num_hidden_layers")) or 0
            ),
            auxiliary_head=bool(
                _positive_int(text.get("mtp_num_hidden_layers"))
            ),
        ),
        text_backbone_only=has_vision,
    )


def _import_qwen35_patterns(
    text: Mapping[str, Any],
    layers: int,
    hidden: int,
    intermediate: int,
    heads: int,
    kv_heads: int,
    experts: int,
    top_k: int,
    *,
    dtype: str,
    quantization: Optional[str],
) -> Tuple[LayerPattern, ...]:
    required = (layers, hidden, intermediate, heads, kv_heads)
    layer_types = text.get("layer_types")
    if not all(required) or not isinstance(layer_types, list) or len(layer_types) != layers:
        return ()
    if any(item not in {"linear_attention", "full_attention"} for item in layer_types):
        return ()
    linear = LinearAttentionPattern(
        key_heads=_positive_int(text.get("linear_num_key_heads")) or 0,
        value_heads=_positive_int(text.get("linear_num_value_heads")) or 0,
        key_head_dim=_positive_int(text.get("linear_key_head_dim")) or 0,
        value_head_dim=_positive_int(text.get("linear_value_head_dim")) or 0,
        conv_kernel_size=_positive_int(text.get("linear_conv_kernel_dim")) or 1,
        state_dtype=_state_dtype(str(text.get("mamba_ssm_dtype") or "fp32")),
        output_gate=bool(text.get("attn_output_gate", True)),
        gate_activation=str(text.get("hidden_act") or "silu"),
    )
    if min(linear.key_heads, linear.value_heads, linear.key_head_dim, linear.value_head_dim) <= 0:
        return ()
    shared = _positive_int(text.get("shared_expert_intermediate_size")) or 0
    kind = "moe" if experts > 1 else "dense"
    patterns: List[LayerPattern] = []
    start = 0
    while start < len(layer_types):
        mixer = layer_types[start]
        end = start + 1
        while end < len(layer_types) and layer_types[end] == mixer:
            end += 1
        patterns.append(
            LayerPattern(
                repeat=end - start,
                kind=kind,
                hidden_size=hidden,
                intermediate_size=intermediate,
                attention_heads=heads,
                kv_heads=kv_heads,
                num_experts=experts,
                experts_per_token=top_k,
                label="{}_block".format(mixer),
                sequence_mixer=mixer,
                linear_attention=linear if mixer == "linear_attention" else None,
                shared_expert_intermediate_size=shared,
                shared_expert_gate=shared > 0,
                attention_head_dim=_positive_int(text.get("head_dim")) or 0,
                dtype=dtype,
                quantization=quantization,
            )
        )
        start = end
    return tuple(patterns)


def _bundled_definitions() -> Iterable[PresetDefinition]:
    from .model_presets import get_model_preset

    for item in list_model_presets():
        yield get_model_preset(item["id"])


def _definition_to_json(definition: PresetDefinition) -> Dict[str, Any]:
    return asdict(definition)


def _definition_from_json(raw: Any) -> PresetDefinition:
    if not isinstance(raw, dict):
        raise ValueError("cached definition must be an object")
    values = dict(raw)
    pattern_values = values.pop("patterns", [])
    evidence = values.get("architecture_evidence")
    if isinstance(evidence, dict):
        values["architecture_evidence"] = ArchitectureEvidence(**evidence)
    mtp = values.get("mtp")
    if isinstance(mtp, dict):
        values["mtp"] = MTPBranchSpec(**mtp)
    patterns = []
    for item in pattern_values:
        if not isinstance(item, dict):
            raise ValueError("cached layer pattern must be an object")
        pattern = dict(item)
        linear = pattern.get("linear_attention")
        if isinstance(linear, dict):
            pattern["linear_attention"] = LinearAttentionPattern(**linear)
        patterns.append(LayerPattern(**pattern))
    values["patterns"] = tuple(patterns)
    for field in (
        "modalities",
        "supported_modalities",
        "unsupported_subgraphs",
        "limitations",
        "variants",
    ):
        values[field] = tuple(values.get(field, ()))
    return PresetDefinition(**values)


def _repository_provenance(info: Mapping[str, Any]) -> Dict[str, Any]:
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    return {
        "id": info.get("id") or info.get("modelId"),
        "author": info.get("author"),
        "sha": info.get("sha"),
        "last_modified": info.get("lastModified"),
        "gated": info.get("gated"),
        "private": info.get("private", False),
        "license": card.get("license") or _license_from_tags(info.get("tags")),
        "license_name": card.get("license_name"),
        "license_link": card.get("license_link"),
    }


def _license_from_info(info: Mapping[str, Any]) -> str:
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    license_id = card.get("license") or _license_from_tags(info.get("tags"))
    license_name = card.get("license_name")
    if license_id == "other" and isinstance(license_name, str) and license_name.strip():
        return license_name.strip()
    return _license_text(license_id)


def _license_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return " OR ".join(items) if items else "unknown"
    return str(value or "unknown").strip() or "unknown"


def _license_from_tags(tags: Any) -> Optional[str]:
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                return tag.split(":", 1)[1]
    return None


def _license_policy(license_id: str) -> Tuple[str, str]:
    normalized = license_id.strip().lower()
    if normalized in _LICENSE_POLICY_ALIASES:
        return _LICENSE_POLICY_ALIASES[normalized]
    if normalized in _OSI_LICENSES:
        return "open_source", "allowed"
    if any(marker in normalized for marker in _NONCOMMERCIAL_LICENSE_MARKERS):
        return "open_weight", "prohibited"
    if normalized in {"unknown", "other", ""}:
        return "open_weight", "unknown"
    return "open_weight", "conditional"


def _only_full_attention(value: Any) -> bool:
    return value is None or (
        isinstance(value, list)
        and all(item == "full_attention" for item in value)
    )


def _first(values: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if values.get(name) is not None:
            return values[name]
    return None


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return result if result > 0 else 0


def _state_dtype(value: str) -> str:
    normalized = value.strip().lower()
    return {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}.get(normalized, normalized)


def _import_precision(
    text: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[str, Optional[str], bool, Tuple[str, ...]]:
    """Read only explicit config precision; never infer it from a repo name."""

    limitations: List[str] = []
    dtype_raw = _first(text, "torch_dtype", "dtype")
    if dtype_raw is None and text is not config:
        dtype_raw = _first(config, "torch_dtype", "dtype")
    dtype, dtype_error = _canonical_import_dtype(dtype_raw)
    supported = dtype_error is None
    if dtype_error is not None:
        limitations.append(dtype_error)
    elif dtype_raw is None or str(dtype_raw).strip().lower() in {"", "auto"}:
        limitations.append(
            "torch_dtype/dtype is absent or auto; BF16 is retained as an explicit analytical precision fallback."
        )

    nested_quantization = (
        text.get("quantization_config") if text is not config else None
    )
    top_quantization = config.get("quantization_config")
    quantization_raw = (
        nested_quantization
        if nested_quantization is not None
        else top_quantization
    )
    if (
        nested_quantization is not None
        and top_quantization is not None
        and nested_quantization != top_quantization
    ):
        quantization = None
        quantization_error = (
            "Nested text_config and top-level quantization_config conflict; the import remains metadata-only."
        )
    else:
        quantization, quantization_error = _import_quantization(
            quantization_raw
        )
    if quantization_error is not None:
        supported = False
        limitations.append(quantization_error)
    elif quantization is not None:
        limitations.append(
            "Explicit quantization_config storage bits were imported as {}; kernel packing, scales/zeros, and accuracy are not inferred."
            .format(quantization)
        )
    return dtype, quantization, supported, tuple(limitations)


def _canonical_import_dtype(value: Any) -> Tuple[str, Optional[str]]:
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return "bf16", None
    if not isinstance(value, str):
        return "bf16", (
            "torch_dtype/dtype must be a supported string; the import remains metadata-only."
        )
    normalized = value.strip().lower()
    if normalized.startswith("torch."):
        normalized = normalized[6:]
    normalized = normalized.replace("-", "").replace("_", "")
    dtype = _IMPORT_DTYPE_ALIASES.get(normalized)
    if dtype is None:
        return "bf16", (
            "Unsupported torch_dtype/dtype {!r}; the import remains metadata-only."
            .format(value)
        )
    return dtype, None


def _import_quantization(
    value: Any,
) -> Tuple[Optional[str], Optional[str]]:
    if value is None or value == {}:
        return None, None
    if not isinstance(value, Mapping):
        return None, (
            "quantization_config must be an object; the import remains metadata-only."
        )

    bit_candidates = set()
    for key in ("bits", "weight_bits", "wbits"):
        if key not in value or value.get(key) is None:
            continue
        raw = value.get(key)
        if isinstance(raw, bool):
            return None, (
                "quantization_config {} must be an integer; the import remains metadata-only."
                .format(key)
            )
        try:
            bits = int(raw)
        except (TypeError, ValueError, OverflowError):
            return None, (
                "quantization_config {} must be an integer; the import remains metadata-only."
                .format(key)
            )
        if str(raw).strip() != str(bits):
            return None, (
                "quantization_config {} must be an exact integer; the import remains metadata-only."
                .format(key)
            )
        bit_candidates.add(bits)
    if value.get("load_in_4bit") is True or value.get("bnb_4bit_quant_type"):
        bit_candidates.add(4)
    if value.get("load_in_8bit") is True:
        bit_candidates.add(8)
    method = str(value.get("quant_method") or "").strip().lower()
    explicit_method_bits = re.search(r"(?:^|[_-])(4|8)bit(?:$|[_-])", method)
    if explicit_method_bits:
        bit_candidates.add(int(explicit_method_bits.group(1)))

    if len(bit_candidates) != 1:
        reason = "conflicting" if len(bit_candidates) > 1 else "ambiguous"
        return None, (
            "quantization_config is {}; an exact supported weight bit-width is required, so the import remains metadata-only."
            .format(reason)
        )
    bits = next(iter(bit_candidates))
    if bits not in {4, 8}:
        return None, (
            "quantization_config weight bits {} are unsupported; the import remains metadata-only."
            .format(bits)
        )
    return "w{}".format(bits), None


def _import_preset_id(repo_id: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", repo_id.lower()).strip("-")
    return "hf-{}".format(value[:120])


def _family_from_name(name: str) -> str:
    match = re.match(r"([A-Za-z]+(?:[._-]?[A-Za-z]+)*?)(?=[._-]?\d|$)", name)
    family = match.group(1) if match else name.split("-", 1)[0]
    return family.replace("_", ".").strip("-.") or "Imported"


def _scale_from_name(name: str) -> str:
    match = re.search(r"(?i)(\d+(?:\.\d+)?[MBT](?:[-_/]?A\d+(?:\.\d+)?[MB])?)", name)
    return match.group(1).upper() if match else "unspecified"


__all__ = [
    "CatalogError",
    "HuggingFaceClient",
    "ModelCatalog",
    "RemoteCatalogError",
    "default_catalog_cache_dir",
    "definition_from_huggingface",
    "validate_repo_id",
    "validate_revision",
]
