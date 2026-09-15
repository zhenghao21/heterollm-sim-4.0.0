"""Prepare exact, frozen native /completion token arrays without model inference.

The tokenizer is the executable adjacent to the selected server; never fall back
silently to a different build.  Lengths count every returned token, including the
model's automatic BOS where applicable.  Models with add_bos=false retain that
native policy.  Submit the stored integer array, not a retokenized text prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "native-grid-prompts/v1"
LENGTHS = (128, 512, 1536)
DEFAULT_PROTOCOL = ROOT / "artifacts/development/native_repeatability_5pct/phase_a_protocol.json"
DEFAULT_RUNTIME = ROOT / "source/llama.cpp-native-thread-control/build-native-thread-control/bin/llama-server.exe"
DEFAULT_OUTPUT = ROOT / "artifacts/development/native_repeatability_5pct/grid_prompt_manifest.json"

# Original, deliberately plain prose shared byte-for-byte across every model.
# These are continuous natural-language observations, not token/character padding.
CORPUS_TEXT = """Read the following field journal about a community research station. Continue the account in the same clear and observational style, explaining the work and the reasons behind each decision.

At the beginning of spring, a small team opened a research station beside a river. The building had once been a village school, and its tall windows admitted the early light. The team wanted to understand how water, soil, plants, and everyday human activity changed together through the year. They began with modest questions that could be answered through careful observation. Which paths became muddy after rain? How long did pools remain in the meadow? Where did young trees survive the summer? A notebook on the kitchen table held their first plans.

Before collecting measurements, the observers walked the valley with residents who knew it well. A farmer described the way water crossed a field after a storm. A teacher remembered when children could cross a shallow channel without getting wet. A carpenter pointed to marks on an old bridge that recorded several high floods. These stories were not treated as precise measurements, but they helped the team choose places to study. The observers wrote down who had described each event and whether the account came from direct experience or from a story passed between neighbors.

The first practical task was to draw a map. The group marked the station, the river, the footpaths, the fields, and the houses. They divided the study area into smaller sections that could be visited within one morning. Each section received a short name that was easy to pronounce over the radio. Two people checked the map by walking the same route in opposite directions. When their descriptions disagreed, they returned together and examined the ground. This slow process prevented confusion later, when measurements from different weeks needed to be compared.

The station stored its instruments in labeled drawers. A thermometer belonged in one drawer, a measuring tape in another, and sample bottles on a shelf near the door. Every instrument had a simple record of its condition. Before a field visit, the observer checked the battery, inspected the casing, and compared the reading with a reference. After the visit, equipment was cleaned and returned to the same place. This routine was less exciting than discovering a new pattern in the river, yet it made the discoveries easier to trust. Missing equipment could otherwise turn a planned morning into wasted travel.

Water level was measured at a fixed post beneath the bridge. The post carried marks that remained visible from a safe position on the bank. Observers recorded the time, the weather, and any unusual conditions such as floating branches or disturbed soil. They avoided changing the reference point simply because another position looked more convenient. When the post required repair, they measured its relation to a second marker before touching it. That extra measurement allowed the old and new records to remain connected without pretending that the physical setup had never changed.

Rainfall presented a different challenge. The garden behind the station seemed like an obvious place for a rain gauge, but a nearby roof could redirect water and a large tree could intercept it. The group compared several locations and selected an open patch of ground. They documented its distance from obstacles and photographed the surroundings. A volunteer emptied the collector at the same time each morning. If the volunteer was absent, the next person noted the longer interval instead of dividing the total into invented daily values. An honest gap was more useful than a convincing fiction.

Each sample bottle carried a label before it entered the field. The label included a location, a date, a sequence number, and the initials of the collector. The team used waterproof ink because damp paper had already spoiled one practice run. Bottles were carried in separate compartments so they would not strike one another on rough paths. At the station, another person compared the labels with the notebook. When a label could not be read, the sample was set aside for review. Nobody guessed its origin merely to keep the table complete.

The observers also studied the meadow. They placed small permanent squares in areas with different amounts of shade and moisture. Within each square they recorded plant cover, exposed soil, and signs of grazing. They tried to describe what they saw without making an explanation part of the observation. A yellow leaf was recorded as yellow before anyone suggested drought or disease. Photographs helped the team revisit uncertain cases. The same camera position and a small scale marker made comparisons clearer, even when the person taking the photograph changed from one week to the next.

By late spring, the notebooks contained many rows of numbers. The group created a shared table but kept the original records. One column contained the observed value, another contained the unit, and a third explained unusual circumstances. A blank cell meant that a measurement was missing; a zero meant that an actual measurement had produced zero. These meanings were written near the top of the table. The distinction seemed obvious during a meeting, but it was easy to forget when copying several pages after a long day outside. Clear definitions reduced the need to rely on memory.

Once a week, two observers reviewed a small selection of entries. They compared the electronic table with the paper notebook and examined values that looked different from nearby observations. A surprising value did not automatically become an error. Sometimes it revealed a real event, such as heavy rain in one part of the valley. At other times it came from a misplaced decimal point or a forgotten unit conversion. Corrections included a note describing the reason and retained a trace of the earlier value. The review was designed to improve the record, not to reward smooth-looking results.

The station welcomed visitors on Saturday afternoons. A large map on the wall showed the study sites, and a display explained how a rain gauge worked. Children were invited to compare stones from different parts of the river, then describe the differences in their own words. Adults often asked whether the work would predict the next flood. The team explained what the observations could support and what remained uncertain. A few months of data could reveal useful local patterns, but it could not replace a long record or a complete understanding of the surrounding hills.

During the first summer heat wave, the meadow changed quickly. Patches that had been green in the morning looked dull by evening. The observers added an extra visit but kept the normal schedule intact so comparisons with earlier weeks remained possible. They recorded the additional visit separately. The river fell below one of its familiar marks, exposing stones that had remained hidden throughout spring. The group photographed the change and noted where shallow water interrupted the usual flow. They resisted the temptation to describe every unusual sight as an unprecedented event.

Working in hot weather required changes in routine. Field visits began earlier, drinking water was carried in separate containers, and difficult routes were assigned to pairs. The team checked in at agreed times and left a copy of each route at the station. If someone returned late, the others knew where to begin looking. These arrangements were described in ordinary language rather than in a long set of rules that nobody could remember. Safety was treated as part of organizing good observations because tired or rushed people were also more likely to make mistakes.

A storm in midsummer provided an unexpected test of the system. Rain arrived before dawn and continued until the afternoon. One route became inaccessible, while another remained safe from higher ground. The observers followed the safe route and marked the other measurements as unavailable. They did not cross moving water to avoid leaving a gap in the record. After the storm, they inspected the fixed markers, checked for shifted soil, and compared fresh photographs with earlier images. The event showed why records of the measurement setup mattered as much as the numbers collected from it.

The team discussed how to summarize its growing collection of observations. An average could describe a typical level, but it could hide the effect of a few extreme days. They therefore displayed the individual observations as well as a summary. The dates of storms and maintenance were shown beside the plot. Missing intervals remained visible. When two sites were compared, both were shown with the same units and a clearly stated time span. Visitors could then distinguish a difference in the river from a difference in the way the information had been displayed.

As autumn approached, the station prepared a report for the village. The report began with the questions that had guided the work, then described the locations and methods. It separated direct observations from possible explanations. The team included a section on failed equipment and missing measurements because these details affected how confidently the results could be used. The final pages proposed a few practical next steps, such as repairing a drainage channel and continuing observations through winter. Each proposal explained the evidence behind it and what further information would help refine the decision.

The closing meeting took place in the old classroom on a rainy evening. Residents studied the maps, compared them with their own experience, and suggested places that deserved attention next year. Some questions remained unanswered, but the group now had a common record from which to begin. Before everyone left, a volunteer checked the rain gauge and entered the reading in the notebook. The small act captured the purpose of the station: useful knowledge grew through repeated, careful work, shared definitions, and a willingness to record uncertainty as faithfully as a clear result.
"""


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def file_identity(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"path": str(path), "size_bytes": after.st_size, "sha256": digest.hexdigest()}


def parse_tokenizer_output(stdout: str, stderr: str = "", returncode: int = 0) -> list[int]:
    matches = re.findall(r"(?m)^\s*(\[[0-9,\s-]*\])\s*$", stdout)
    counts = re.findall(r"Total number of tokens:\s*(\d+)", stdout + "\n" + stderr)
    if returncode or len(matches) != 1 or len(counts) != 1:
        raise RuntimeError(f"invalid tokenizer response (exit={returncode}, ids={len(matches)}, counts={len(counts)})")
    ids = json.loads(matches[0])
    if any(type(i) is not int or i < 0 for i in ids) or len(ids) != int(counts[0]):
        raise RuntimeError("tokenizer ID/count mismatch or invalid token ID")
    return ids


def tokenize(executable: Path, model: Path, text: str, *, no_bos: bool = False) -> dict:
    argv = [str(executable), "-m", str(model), "--stdin", "--ids", "--show-count", "--no-escape", "--no-parse-special"]
    if no_bos:
        argv.append("--no-bos")
    result = subprocess.run(argv, input=text.encode("utf-8"), capture_output=True, timeout=180, check=False)
    stdout = result.stdout.decode("utf-8", errors="strict")
    stderr = result.stderr.decode("utf-8", errors="replace")
    ids = parse_tokenizer_output(stdout, stderr, result.returncode)
    return {"ids": ids, "count": len(ids), "argv": argv, "stdin_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(), "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()}


def token_prefixes(ids: list[int], no_bos_ids: list[int], lengths: tuple[int, ...] = LENGTHS) -> tuple[dict, dict]:
    if not ids or any(type(i) is not int or i < 0 for i in ids + no_bos_ids):
        raise ValueError("nonempty integer token arrays required")
    if ids == no_bos_ids:
        bos = {"policy": "model_native", "automatically_added": False, "token_id": None, "tokens_in_budget": 0}
    elif len(ids) == len(no_bos_ids) + 1 and ids[1:] == no_bos_ids:
        bos = {"policy": "model_native", "automatically_added": True, "token_id": ids[0], "tokens_in_budget": 1}
    else:
        raise ValueError("unexpected special-token difference between default and --no-bos")
    if not lengths or any(type(n) is not int or n < 1 for n in lengths) or len(set(lengths)) != len(lengths):
        raise ValueError("lengths must be distinct positive integers")
    if len(ids) < max(lengths):
        raise ValueError(f"corpus has only {len(ids)} actual tokens; need {max(lengths)}")
    prompts = {}
    for count in lengths:
        prefix = ids[:count]
        prompts[str(count)] = {"count": count, "ids": prefix, "prompt_token_ids": prefix, "expected_prompt_tokens": count, "ids_sha256": stable_hash(prefix),
                               "source_token_range": [0, count], "source": "corpus.text", "bos_tokens_in_budget": bos["tokens_in_budget"]}
    return prompts, bos


def server_contract(root: Path = ROOT) -> dict:
    common = root / "source/llama.cpp-semantic-patches/tools/server/server-common.cpp"
    context = root / "source/llama.cpp-annotation-control/tools/server/server-context.cpp"
    route = root / "source/llama.cpp-semantic/tools/server/server.cpp"
    tokenizer = root / "source/llama.cpp-semantic/tools/tokenize/tokenize.cpp"
    checks = [
        (common, "llama_tokens tmp = json_prompt.get<llama_tokens>();", "Integer arrays are copied directly into server_tokens."),
        (common, "prompt_tokens.push_back(p.template get<llama_token>());", "Numeric members in mixed prompts are appended without tokenization."),
        (context, "inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);", "The native completion handler uses the shared prompt parser."),
        (route, 'ctx_http.post("/completion",', "The native /completion route selects post_completions."),
        (tokenizer, "model_params.vocab_only = true;", "The tokenizer loads vocabulary only, not model weights."),
        (tokenizer, "const bool add_bos      = model_wants_add_bos && !params.tokenize_no_bos;", "Automatic BOS follows the model policy and can be disabled for verification."),
    ]
    evidence = []
    for path, needle, meaning in checks:
        lines = path.read_text(encoding="utf-8").splitlines()
        locations = [i + 1 for i, line in enumerate(lines) if needle in line]
        if len(locations) != 1:
            raise RuntimeError(f"source contract anchor missing or ambiguous: {path}: {needle}")
        line = locations[0]
        evidence.append({**file_identity(path), "line": line, "excerpt": "\n".join(lines[max(0, line-3):line+3]), "meaning": meaning})
    return {"endpoint": "/completion", "prompt_type": "array_of_integer_token_ids", "retokenizes": False,
            "adds_bos_to_integer_array": False, "count_includes_native_bos": True, "evidence": evidence}


def prepare_manifest(protocol_path: Path = DEFAULT_PROTOCOL, runtime: Path = DEFAULT_RUNTIME, *,
                     corpus_text: str = CORPUS_TEXT, lengths: tuple[int, ...] = LENGTHS,
                     tokenizer_fn: Callable = tokenize, capture_contract: bool = True) -> dict:
    runtime = runtime.resolve(strict=True)
    executable = runtime.with_name("llama-tokenize.exe").resolve(strict=True)
    protocol_path = protocol_path.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    files = [file_identity(runtime), file_identity(executable)]
    files.extend(file_identity(p) for p in sorted(runtime.parent.glob("*.dll")))
    receipts = [file_identity(p) for p in [
        ROOT / "source/llama.cpp-native-thread-control/evidence/build_receipt.json",
        ROOT / "source/llama.cpp-native-thread-control/evidence/source_manifest.json",
        ROOT / "source/llama.cpp-annotation-control/evidence/build_receipt.json",
        ROOT / "source/llama.cpp-annotation-control/evidence/source_manifest.json",
    ] if p.exists()]
    models = []
    seen = set()
    for job in protocol["jobs"]:
        model = Path(job["model"]).resolve(strict=True)
        if model in seen:
            continue
        seen.add(model)
        identity = file_identity(model)
        first = tokenizer_fn(executable, model, corpus_text)
        second = tokenizer_fn(executable, model, corpus_text)
        without = tokenizer_fn(executable, model, corpus_text, no_bos=True)
        if first["ids"] != second["ids"]:
            raise RuntimeError(f"tokenizer repeat verification failed: {model}")
        prompts, bos = token_prefixes(first["ids"], without["ids"], lengths)
        models.append({"model_id": job["id"].split("_")[0], "source_job_id": job["id"], "model": str(model),
                       "model_sha256": identity["sha256"], "model_identity": identity, "bos": bos, "prompts": prompts,
                       "source_token_ids": first["ids"], "source_token_ids_sha256": stable_hash(first["ids"]),
                       "source_token_count": len(first["ids"]),
                       "verification": {"repeat_ids_equal": True, "no_bos_comparison": True,
                                        "runs": [{k: v for k, v in r.items() if k != "ids"} for r in (first, second, without)]}})
    if not models:
        raise ValueError("protocol contains no models")
    manifest = {"schema": SCHEMA, "lengths": list(lengths), "protocol": file_identity(protocol_path),
                "corpus": {"id": "original-river-research-journal-v1", "origin": "original prose embedded in prepare_native_grid_prompts.py",
                           "text": corpus_text, "encoding": "UTF-8", "sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()},
                "tokenizer": file_identity(executable), "runtime": {"server": str(runtime), "files": files, "sha256": stable_hash(files), "build_evidence": receipts},
                "special_tokens": {"policy": "model_native_bos", "parse_special": False,
                                   "note": "Exact lengths include an automatic BOS when the model uses one; no BOS is invented for models with add_bos=false."},
                "server_contract": server_contract() if capture_contract else None, "models": models}
    manifest["manifest_sha256"] = stable_hash(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported prompt manifest schema")
    basis = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != stable_hash(basis):
        raise ValueError("prompt manifest checksum mismatch")
    corpus = manifest["corpus"]
    if hashlib.sha256(corpus["text"].encode("utf-8")).hexdigest() != corpus["sha256"]:
        raise ValueError("corpus checksum mismatch")
    for model in manifest["models"]:
        ids = model["source_token_ids"]
        if stable_hash(ids) != model["source_token_ids_sha256"] or len(ids) != model["source_token_count"]:
            raise ValueError("source token checksum/count mismatch")
        for count in manifest["lengths"]:
            prompt = model["prompts"][str(count)]
            if prompt["count"] != count or prompt.get("expected_prompt_tokens") != count or prompt.get("prompt_token_ids") != prompt["ids"] or len(prompt["ids"]) != count or prompt["ids"] != ids[:count]:
                raise ValueError("prompt is not the exact source token prefix")
            if prompt["ids_sha256"] != stable_hash(prompt["ids"]):
                raise ValueError("prompt token checksum mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--exe", "--runtime", dest="runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus", type=Path, help="Optional UTF-8 natural prose; frozen verbatim in the manifest")
    args = parser.parse_args()
    corpus = args.corpus.read_text(encoding="utf-8") if args.corpus else CORPUS_TEXT
    manifest = prepare_manifest(args.protocol, args.runtime, corpus_text=corpus)
    validate_manifest(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.out.resolve()), "manifest_sha256": manifest["manifest_sha256"],
                      "models": [{"id": m["model_id"], "source_tokens": m["source_token_count"], "bos": m["bos"],
                                  "prompt_counts": [p["count"] for p in m["prompts"].values()]} for m in manifest["models"]]}, indent=2))


if __name__ == "__main__":
    main()
