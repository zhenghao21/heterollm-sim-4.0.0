"""Render heatmaps for the frozen five-model generalization acceptance report."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from PIL import Image, ImageDraw

MODELS = ("qwen25", "qwen35", "qwen38", "tinyllama", "smollm2")
PROMPTS = ("short", "medium", "long")
OUTPUTS = ("short", "medium", "long")
PARALLEL = (1, 2, 4)
METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")
STATS_PCT = ("median_of_repeats_abs_pct", "p90_abs_pct", "worst_abs_pct")
STATS_MS = ("median_absolute_ms", "p90_absolute_ms", "worst_absolute_ms")

def scale_color(v, vmax):
    if v is None: return (225, 225, 225)
    t = max(0.0, min(1.0, float(v) / vmax if vmax else 0.0))
    return (255, int(250 - 175*t), int(235 - 190*t))

def render(report, out_dir, metric, stat):
    agg = report.get("aggregation", {})
    cols = [(o, p) for o in OUTPUTS for p in PARALLEL]
    rows = [(m, pr) for m in MODELS for pr in PROMPTS]
    vals = []
    for m, pr in rows:
        for o, p in cols:
            obj = agg.get(f"{m}|{pr}|{o}|{p}", {}).get("metrics", {}).get(metric, {})
            v = obj.get(stat)
            if isinstance(v, (int, float)) and math.isfinite(float(v)): vals.append(float(v))
    vmax = max(vals) if vals else 1.0
    if vmax <= 0: vmax = 1.0
    width, left, cw, rh, top = 1400, 170, 132, 30, 58
    height = top + len(rows)*rh + 70
    img = Image.new("RGB", (width, height), "white"); d = ImageDraw.Draw(img)
    unit = "%" if stat.endswith("pct") else "ms"
    label = "DEVELOPMENT SCREENING" if report.get("independent_blind") is False else "SCREENING / NOT A BLIND ACCEPTANCE PASS"
    d.text((20, 10), f"L1 ENGINE {metric.upper()} {stat} ({unit}) | {label}", fill="black")
    for j, (o, p) in enumerate(cols):
        d.text((left + j*cw + 24, 35), f"{o}/p{p}", fill="black")
    for i, (m, pr) in enumerate(rows):
        y = top + i*rh
        d.text((20, y+8), f"{m}/{pr}", fill="black")
        for j, (o, p) in enumerate(cols):
            obj = agg.get(f"{m}|{pr}|{o}|{p}", {}).get("metrics", {}).get(metric, {})
            v = obj.get(stat); v = float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None
            x = left + j*cw
            d.rectangle((x, y, x+cw-4, y+rh-2), fill=scale_color(v, vmax), outline="black")
            text = ("not planned" if f"{m}|{pr}|{o}|{p}" not in agg else "insufficient") if v is None else f"{v:.1f}{unit}"
            d.text((x+cw//2-22, y+8), text, fill="black")
    d.text((left, height-30), f"0 {unit}", fill="black")
    d.text((width-120, height-30), f"max {vmax:.1f}{unit}", fill="black")
    d.text((20, height-14), "Gray: not planned / insufficient. P90 and worst panels summarize repeat errors per scenario; model-group scenario P90 is in the report.", fill="black")
    out = out_dir / f"{metric}_{stat}.png"; img.save(out); return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(); report = json.loads(a.report.read_text(encoding="utf-8")); a.out_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        for stat in STATS_PCT + STATS_MS:
            print(render(report, a.out_dir, metric, stat))

if __name__ == "__main__": main()
