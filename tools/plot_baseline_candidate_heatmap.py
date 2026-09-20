"""Render a baseline/candidate APE heatmap for the frozen score schema."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

METRICS = ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms")
LABELS = {"engine_ttft_ms": "TTFT", "engine_tpot_ms": "TPOT", "engine_e2e_ms": "E2E"}
NO_CANDIDATE = "NO CANDIDATE"


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def read_score(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, float]]]:
    path = path.resolve(strict=True)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "stable-native-simulation-errors/v2":
        raise ValueError(f"unsupported score schema: {path}")
    cells = document.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"score cells missing: {path}")
    values: dict[tuple[str, str], dict[str, float]] = {}
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("cell_id"), str) or not isinstance(cell.get("model_key"), str):
            raise ValueError(f"malformed score cell: {path}")
        key = (cell["model_key"], cell["cell_id"])
        if key in values:
            raise ValueError(f"duplicate score cell: {key}")
        metrics = cell.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"score metrics missing: {key}")
        parsed: dict[str, float] = {}
        for metric in METRICS:
            value = (metrics.get(metric) or {}).get("absolute_percentage_error_pct")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"non-finite {metric} APE: {key}")
            parsed[metric] = float(value)
        values[key] = parsed
    return document, values


def _rgb(value: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % value


def _ape_color(value: float | None) -> tuple[int, int, int]:
    if value is None:
        return 224, 224, 224
    # Green is below the 25% gate; red intensity grows with the error.
    ratio = max(0.0, min(1.0, value / 100.0))
    if value < 25.0:
        t = max(0.0, min(1.0, value / 25.0))
        return int(218 + 25 * t), int(245 - 35 * t), int(218 - 35 * t)
    t = max(0.0, min(1.0, (value - 25.0) / 75.0))
    return 255, int(218 - 150 * t), int(218 - 150 * t)


def _delta_color(value: float | None) -> tuple[int, int, int]:
    if value is None:
        return 224, 224, 224
    t = max(-1.0, min(1.0, value / 50.0))
    if t < 0:
        q = -t
        return int(218 - 65 * q), int(238 - 35 * q), 255
    return 255, int(238 - 75 * t), int(218 - 75 * t)


def render_heatmap(rows: list[tuple[str, str]], baseline: dict[tuple[str, str], dict[str, float]],
                  candidate: dict[tuple[str, str], dict[str, float]] | None, metric: str, output: Path) -> None:
    row_height, label_width, cell_width = 20, 310, 190
    width = label_width + cell_width * 3 + 30
    height = 70 + row_height * len(rows)
    title = f"{LABELS[metric]} absolute percentage error | baseline vs candidate"
    headers = ("BASELINE APE", "CANDIDATE APE" if candidate is not None else NO_CANDIDATE, "DELTA (CANDIDATE - BASELINE)")
    shapes = [f'<rect width="{width}" height="{height}" fill="white"/>',
              f'<text x="12" y="20" font-family="Arial" font-size="14">{html.escape(title)}</text>']
    for index, header in enumerate(headers):
        shapes.append(f'<text x="{label_width + index * cell_width + 6}" y="52" font-family="Arial" font-size="11">{html.escape(header)}</text>')
    for index, (model_key, cell_id) in enumerate(rows):
        y = 62 + index * row_height
        label = html.escape(f"{model_key}/{cell_id}")
        shapes.append(f'<text x="8" y="{y + 14}" font-family="Arial" font-size="11">{label}</text>')
        base_value = baseline[(model_key, cell_id)][metric]
        candidate_value = candidate[(model_key, cell_id)][metric] if candidate is not None else None
        values = (base_value, candidate_value, candidate_value - base_value if candidate_value is not None else None)
        for column, value in enumerate(values):
            x = label_width + column * cell_width
            fill = _delta_color(value) if column == 2 else _ape_color(value)
            shapes.append(f'<rect x="{x}" y="{y}" width="{cell_width - 8}" height="{row_height - 2}" fill="{_rgb(fill)}" stroke="#888888" stroke-width="1"/>')
            text = "—" if value is None else f"{value:+.1f}%" if column == 2 else f"{value:.1f}%"
            shapes.append(f'<text x="{x + 8}" y="{y + 14}" font-family="Arial" font-size="11">{html.escape(text)}</text>')
    legend_y = height - 8
    output.parent.mkdir(parents=True, exist_ok=True)
    shapes.append(f'<text x="8" y="{legend_y}" font-family="Arial" font-size="10">green &lt;25% gate | red &gt;=25% | delta blue=improvement, red=regression | gray=missing</text>')
    output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">' + "".join(shapes) + "</svg>\n", encoding="utf-8")


def generate(baseline_path: Path, output_dir: Path, candidate_path: Path | None = None) -> dict[str, Any]:
    baseline_document, baseline = read_score(baseline_path)
    candidate_document = None
    candidate = None
    if candidate_path is not None:
        candidate_document, candidate = read_score(candidate_path)
        if set(candidate) != set(baseline):
            raise ValueError("candidate score cell set differs from baseline")
    rows = sorted(baseline, key=lambda key: (key[0], key[1]))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for metric in METRICS:
        path = output_dir / f"{metric}_baseline_candidate.svg"
        render_heatmap(rows, baseline, candidate, metric, path)
        artifacts.append(file_ref(path))
    manifest = {
        "schema": "baseline-candidate-error-heatmap/v1",
        "baseline_score_ref": file_ref(baseline_path),
        "candidate_score_ref": file_ref(candidate_path) if candidate_path is not None else None,
        "candidate_status": "present" if candidate is not None else "absent",
        "rows": len(rows), "metrics": list(METRICS),
        "threshold_pct_strict": 25, "comparison": "candidate_ape_minus_baseline_ape",
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--candidate-score", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = generate(args.baseline_score, args.out_dir, args.candidate_score)
    print(manifest["manifest_path"])


if __name__ == "__main__":
    main()
