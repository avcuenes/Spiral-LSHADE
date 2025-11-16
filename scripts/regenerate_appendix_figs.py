#!/usr/bin/env python3
"""
Regenerate the `appendix_figs` folders from the CSV artifacts stored in
`results_*` directories. This lets you rebuild all paper plots without
rerunning any benchmark sweeps.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]


def discover_default_dirs() -> List[Path]:
    """Return results directories that contain the plotting scripts."""
    dirs = []
    for candidate in sorted(ROOT.glob("results_*")):
        if not candidate.is_dir():
            continue
        if not list(candidate.glob("*full_stats.py")):
            continue
        if not list(candidate.glob("*results.csv")):
            continue
        dirs.append(candidate)
    return dirs


def infer_dims_from_name(name: str) -> Optional[str]:
    match = re.search(r"_(\d+)D", name, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def infer_dims_from_csv(csv_path: Path, sample_rows: int = 1000) -> Optional[str]:
    dims = []
    try:
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            headers = [h.lower() for h in (reader.fieldnames or [])]
            if "dim" not in headers:
                return None
            for i, row in enumerate(reader):
                raw = row.get("dim") or row.get("DIM") or ""
                raw = raw.strip()
                if raw:
                    try:
                        dims.append(int(float(raw)))
                    except ValueError:
                        continue
                if i + 1 >= sample_rows:
                    break
    except FileNotFoundError:
        return None

    uniq = sorted(set(dims))
    if not uniq:
        return None
    return ",".join(str(d) for d in uniq)


def infer_dims(results_dir: Path, csv_path: Path) -> Optional[str]:
    return infer_dims_from_name(results_dir.name) or infer_dims_from_csv(csv_path)


def find_unique(results_dir: Path, pattern: str, description: str, required: bool = True) -> Optional[Path]:
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        if required:
            raise FileNotFoundError(f"{description} not found in {results_dir}")
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous {description} in {results_dir}: {[m.name for m in matches]}")
    return matches[0]


def rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def run_command(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print(f"[run] ({cwd}) {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def regenerate_for_dir(results_dir: Path, args: argparse.Namespace) -> None:
    stats_script = find_unique(results_dir, "*full_stats.py", "full-stats script", required=not args.skip_stats)
    mean_script = find_unique(results_dir, "*mean_std.py", "mean/std script", required=not args.skip_mean_std)
    results_csv = find_unique(results_dir, "*results.csv", "results CSV")
    ert_csv = find_unique(results_dir, "*ERT*_summary.csv", "ERT summary CSV", required=False)

    dims = args.dims or infer_dims(results_dir, results_csv)
    if dims:
        print(f"[info] Using dims='{dims}' for {results_dir.name}")

    out_dir = results_dir / "appendix_figs"
    out_rel = rel(out_dir, results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_stats and stats_script is not None:
        cmd = [
            args.python,
            rel(stats_script, results_dir),
            "--csv",
            rel(results_csv, results_dir),
            "--out",
            out_rel,
            "--perf-profile-penalty",
            str(args.perf_profile_penalty),
        ]
        if dims:
            cmd += ["--dims", dims]
        if ert_csv:
            cmd += ["--ert", rel(ert_csv, results_dir)]
        if args.strict_friedman:
            cmd.append("--strict-friedman")
        run_command(cmd, cwd=results_dir, dry_run=args.dry_run)

    if not args.skip_mean_std and mean_script is not None:
        cmd = [
            args.python,
            rel(mean_script, results_dir),
            "--csv",
            rel(results_csv, results_dir),
            "--out",
            out_rel,
            "--metric",
            args.metric,
        ]
        if dims:
            cmd += ["--dims", dims]
        if args.split_by_dim:
            cmd.append("--split-by-dim")
        run_command(cmd, cwd=results_dir, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Regenerate appendix_figs directly from stored results CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "results_dirs",
        nargs="*",
        help="Result directories to process (defaults to all results_* folders that contain plotting scripts).",
    )
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to invoke.")
    ap.add_argument("--dims", help="Override dimension list passed to the plotting scripts.")
    ap.add_argument("--metric", default="err", help="Metric column for mean/std visualizations.")
    ap.add_argument("--perf-profile-penalty", type=float, default=2.0,
                    help="Penalty multiplier for the ERT performance profile.")
    ap.add_argument("--split-by-dim", action="store_true", help="Request per-dimension tables/plots as well.")
    ap.add_argument("--strict-friedman", action="store_true",
                    help="Pass --strict-friedman to the stats script.")
    ap.add_argument("--skip-stats", action="store_true", help="Skip running the full stats script.")
    ap.add_argument("--skip-mean-std", action="store_true", help="Skip running the mean/std script.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.results_dirs:
        dirs = [Path(p).expanduser().resolve() for p in args.results_dirs]
    else:
        dirs = discover_default_dirs()
    if not dirs:
        raise SystemExit("No results directories found. Pass them explicitly, e.g. "
                         "python scripts/regenerate_appendix_figs.py results_cec22_10D")

    for rdir in dirs:
        if not rdir.exists():
            raise SystemExit(f"Results directory '{rdir}' does not exist.")
        print(f"[info] Processing {rdir}")
        regenerate_for_dir(rdir, args)


if __name__ == "__main__":
    main()
