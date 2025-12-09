import ast
import json
import statistics
import subprocess
import sys
from pathlib import Path


def collect_timings(minutes: int = 2) -> list[dict[str, float]]:
    cmd = [
        "docker",
        "compose",
        "-f",
        "docker-compose.yaml",
        "logs",
        "inference",
        "--since",
        f"{minutes}m",
    ]
    raw = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    timings: list[dict[str, float]] = []
    for line in raw.splitlines():
        if "inference_stage_timings" not in line:
            continue
        idx = line.find("{")
        if idx == -1:
            continue
        try:
            payload = ast.literal_eval(line[idx:])
        except Exception:  # pragma: no cover - best effort
            continue
        timings.append({k: float(v) for k, v in payload.get("timings_ms", {}).items()})
    return timings


def summarize(timings: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not timings:
        return {}
    stages = set().union(*(t.keys() for t in timings))
    summary: dict[str, dict[str, float]] = {}
    for stage in sorted(stages):
        values = [t.get(stage, 0.0) for t in timings]
        values_sorted = sorted(values)
        p95_index = min(len(values_sorted) - 1, int(0.95 * (len(values_sorted) - 1)))
        summary[stage] = {
            "mean_ms": round(statistics.fmean(values), 3),
            "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(values_sorted[p95_index], 3),
        }
    return summary


def main() -> None:
    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    timings = collect_timings(minutes)
    print(f"samples={len(timings)}")
    summary = summarize(timings)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
