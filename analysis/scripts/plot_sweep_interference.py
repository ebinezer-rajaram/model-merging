#!/usr/bin/env python3
"""
Plot interference delta vs lambda from the most recent sweep JSON.

By default, this script finds the newest file matching sweep_*.json in the given
--sweeps-dir, then plots:
  - min_interference_delta

It saves a PNG to --output-dir and prints a small summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import tempfile


SWEEP_NAME_RE = re.compile(r"^sweep_(\d{8})_(\d{6})\.json$")


@dataclass(frozen=True)
class SweepPoint:
    lambda_value: float
    min_interference_delta: float | None
    mean_interference_delta: float | None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_sweep_name_timestamp(path: Path) -> datetime | None:
    match = SWEEP_NAME_RE.match(path.name)
    if not match:
        return None
    date_part, time_part = match.groups()
    try:
        return datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def find_latest_sweep(sweeps_dir: Path) -> Path:
    if not sweeps_dir.exists():
        raise FileNotFoundError(f"Sweeps dir not found: {sweeps_dir}")
    candidates = sorted([p for p in sweeps_dir.iterdir() if p.is_file() and p.suffix == ".json"])
    if not candidates:
        raise FileNotFoundError(f"No JSON sweep files found in: {sweeps_dir}")

    def sort_key(path: Path) -> tuple[int, float]:
        ts = _parse_sweep_name_timestamp(path)
        if ts is not None:
            return (1, ts.timestamp())
        return (0, path.stat().st_mtime)

    return max(candidates, key=sort_key)


def extract_points(sweep: dict) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for run in sweep.get("runs", []):
        params = run.get("params") or {}
        lambda_value = params.get("lambda")
        if lambda_value is None:
            lambda_value = params.get("scale")
        if lambda_value is None:
            continue

        score_details = run.get("score_details") or {}
        min_delta = score_details.get("min_interference_delta")
        mean_delta = score_details.get("mean_interference_delta")

        # Fallbacks (some older sweeps may only store `score`).
        if min_delta is None and isinstance(run.get("score"), (int, float)):
            min_delta = float(run["score"])

        # If mean is not explicitly stored, derive it from per-task interference deltas.
        if mean_delta is None:
            per_task = []
            results = run.get("results") or {}
            if isinstance(results, dict):
                for metrics in results.values():
                    if not isinstance(metrics, dict):
                        continue
                    delta = metrics.get("interference_delta")
                    if isinstance(delta, (int, float)):
                        per_task.append(float(delta))
            if per_task:
                mean_delta = sum(per_task) / len(per_task)

        points.append(
            SweepPoint(
                lambda_value=float(lambda_value),
                min_interference_delta=None if min_delta is None else float(min_delta),
                mean_interference_delta=None if mean_delta is None else float(mean_delta),
            )
        )

    return sorted(points, key=lambda p: p.lambda_value)


def _pick_best_lambda(points: list[SweepPoint]) -> float | None:
    scored = [(p.lambda_value, p.min_interference_delta) for p in points if p.min_interference_delta is not None]
    if not scored:
        return None
    # Higher is better for "interference delta" in this repo's sweep scoring.
    return max(scored, key=lambda item: item[1])[0]


def _pick_reference_point(points: list[SweepPoint], reference_lambda: float | None) -> SweepPoint | None:
    if not points:
        return None
    if reference_lambda is None:
        best_lambda = _pick_best_lambda(points)
        if best_lambda is None:
            return None
        for point in points:
            if point.lambda_value == best_lambda:
                return point
        return None
    return min(points, key=lambda point: abs(point.lambda_value - reference_lambda))


def _plot_with_gnuplot(
    points: list[SweepPoint], *, title: str, output_path: Path, reference_lambda: float | None = None
) -> None:
    if not shutil_which("gnuplot"):
        raise RuntimeError("gnuplot not found on PATH; cannot render plot.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_lines = ["lambda\tmin_interference_delta\tmean_interference_delta"]
    for p in points:
        min_value = "" if p.min_interference_delta is None else f"{p.min_interference_delta:g}"
        mean_value = "" if p.mean_interference_delta is None else f"{p.mean_interference_delta:g}"
        data_lines.append(
            f"{p.lambda_value:g}\t"
            f"{min_value}\t"
            f"{mean_value}"
        )
    data_text = "\n".join(data_lines) + "\n"

    best_min_x = _pick_best_lambda(points)
    best_min_point = None
    if best_min_x is not None:
        best_min_y = next(p.min_interference_delta for p in points if p.lambda_value == best_min_x)
        best_min_point = (best_min_x, best_min_y)

    reference_point = _pick_reference_point(points, reference_lambda)

    x_min = min(p.lambda_value for p in points)
    x_max = max(p.lambda_value for p in points)
    tick_step = 0.1
    tick_start = (int(x_min / tick_step)) * tick_step
    if tick_start < x_min:
        tick_start = round(tick_start + tick_step, 10)

    # Use temporary files for gnuplot inputs.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        data_path = tmpdir_path / "data.tsv"
        data_path.write_text(data_text, encoding="utf-8")

        best_min_path = None
        if best_min_point is not None:
            best_min_path = tmpdir_path / "best_min.tsv"
            best_min_path.write_text(f"{best_min_point[0]:g}\t{best_min_point[1]:g}\n", encoding="utf-8")

        guide_cmds: list[str] = []
        if reference_point is not None:
            ref_x = reference_point.lambda_value
            if reference_point.min_interference_delta is not None:
                ref_min = reference_point.min_interference_delta
                guide_cmds.append(
                    f"set arrow 10 from {x_min:g},{ref_min:g} to {ref_x:g},{ref_min:g} "
                    "nohead dt 3 lw 1 lc rgb '#cc3d3d' front"
                )
            if reference_point.mean_interference_delta is not None:
                ref_mean = reference_point.mean_interference_delta
                guide_cmds.append(
                    f"set arrow 11 from {x_min:g},{ref_mean:g} to {ref_x:g},{ref_mean:g} "
                    "nohead dt 3 lw 1 lc rgb '#cc3d3d' front"
                )
            ref_max = max(
                value
                for value in [reference_point.min_interference_delta, reference_point.mean_interference_delta]
                if value is not None
            )
            guide_cmds.append(
                f"set arrow 12 from {x_min:g},{ref_max:g} to {ref_x:g},{ref_max:g} "
                "nohead dt 2 lw 1.4 lc rgb '#b22222' front"
            )
            guide_cmds.append(
                f"set arrow 13 from {ref_x:g},0 to {ref_x:g},1 "
                "nohead dt 2 lw 1.4 lc rgb '#b22222' front"
            )
            guide_cmds.append(
                f"set label 2 sprintf('\\u03bb=%.3g, max=%.3f',{ref_x:g},{ref_max:g}) "
                f"at {ref_x:g},{min(0.98, ref_max + 0.06):g} tc rgb '#b22222' font 'DejaVu Sans,8' center front"
            )

        plot_cmds: list[str] = []
        plot_cmds.append(
            f"'{data_path.as_posix()}' using 1:2 with linespoints ls 1 title 'Min interference delta'"
        )
        plot_cmds.append(
            f"'{data_path.as_posix()}' using 1:3 with linespoints ls 3 title 'Mean interference delta'"
        )
        if best_min_path is not None:
            plot_cmds.append(
                f"'{best_min_path.as_posix()}' using 1:2 with points ls 2 title sprintf('Best \u03bb=%.3g',{best_min_point[0]:g})"
            )

        if not plot_cmds:
            raise ValueError("No plottable interference-delta values found in sweep runs.")

        # Put title in a screen-space label so it doesn't fight with the legend.
        safe_title = title.replace("\n", "\\n").replace('"', '\\"')
        gnuplot_script = "\n".join(
            [
                "set terminal pngcairo size 900,500 font 'DejaVu Sans,10' noenhanced",
                f"set output '{output_path.as_posix()}'",
                "set datafile separator '\\t'",
                "unset title",
                f'set label 1 "{safe_title}" at screen 0.5,0.98 center front',
                "set key at screen 0.5,0.94 center top horizontal",
                "set key maxrows 1",
                "set key font 'DejaVu Sans,9'",
                "set key samplen 1.6",
                "set key spacing 1.0",
                "set key noopaque",
                "set mxtics 2",
                "set mytics 2",
                "set grid xtics ytics lc rgb '#8e8e8e' lw 1.5 dashtype 1",
                "set grid mxtics mytics lc rgb '#c7c7c7' lw 1 dashtype 3",
                f"set xrange [{x_min:g}:{x_max:g}]",
                "set yrange [0:1]",
                f"set xtics {tick_start:g},{tick_step:g}",
                "set ytics 0,0.1,1",
                "set xlabel 'λ'",
                "set ylabel 'Interference delta'",
                "set tmargin 4",
                "set style line 1 lc rgb '#1f77b4' lw 2 pt 7 ps 0.9",
                "set style line 2 lc rgb '#2ca02c' pt 13 ps 1.8",
                "set style line 3 lc rgb '#ff7f0e' lw 2 pt 9 ps 0.9",
                *guide_cmds,
                "plot " + ", \\\n+     ".join(plot_cmds),
                "",
            ]
        )

        script_path = tmpdir_path / "plot.gp"
        script_path.write_text(gnuplot_script, encoding="utf-8")
        subprocess.run(["gnuplot", script_path.as_posix()], check=True)


def _plot_with_matplotlib(
    points: list[SweepPoint], *, title: str, output_path: Path, reference_lambda: float | None = None
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is not installed in this environment. "
            "Install it or use --backend gnuplot."
        ) from exc

    xs = [p.lambda_value for p in points]
    ys_min = [p.min_interference_delta for p in points]
    ys_mean = [p.mean_interference_delta for p in points]
    if not any(v is not None for v in ys_min) and not any(v is not None for v in ys_mean):
        raise ValueError("No plottable interference-delta values found in sweep runs.")

    best_x = _pick_best_lambda(points)
    best_y = None
    if best_x is not None:
        best_y = next(p.min_interference_delta for p in points if p.lambda_value == best_x)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    line = ax.plot(
        xs,
        ys_min,
        marker="o",
        markersize=4,
        linewidth=2.2,
        label="Min interference delta",
        color="#1f77b4",
        alpha=0.95,
    )[0]
    legend_handles = [line]
    legend_labels = ["Min interference delta"]
    if any(v is not None for v in ys_mean):
        line_mean = ax.plot(
            xs,
            ys_mean,
            marker="o",
            markersize=4,
            linewidth=2.2,
            label="Mean interference delta",
            color="#ff7f0e",
            alpha=0.95,
        )[0]
        legend_handles.append(line_mean)
        legend_labels.append("Mean interference delta")
    if best_x is not None and best_y is not None:
        best = ax.scatter(
            [best_x],
            [best_y],
            marker="D",
            s=55,
            label=f"Best λ={best_x:.3g}",
            color="#2ca02c",
            alpha=0.95,
        )
        legend_handles.append(best)
        legend_labels.append(f"Best λ={best_x:.3g}")

    ax.set_title("")
    ax.set_xlabel("λ")
    ax.set_ylabel("Interference delta")
    ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.55, color="#bdbdbd")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.35, color="#d9d9d9")

    x_min, x_max = min(xs), max(xs)
    ax.set_xlim(x_min, x_max)

    reference_point = _pick_reference_point(points, reference_lambda)
    if reference_point is not None:
        ref_x = reference_point.lambda_value
        guide_alpha = 0.45
        guide_color = "#b22222"
        ax.axvline(ref_x, color=guide_color, linestyle="--", linewidth=1.4, alpha=guide_alpha, zorder=1)
        guide_values = []
        if reference_point.min_interference_delta is not None:
            guide_values.append(reference_point.min_interference_delta)
            ax.hlines(
                reference_point.min_interference_delta,
                xmin=x_min,
                xmax=ref_x,
                colors=guide_color,
                linestyles="--",
                linewidth=1.0,
                alpha=guide_alpha,
                zorder=1,
            )
        if reference_point.mean_interference_delta is not None:
            guide_values.append(reference_point.mean_interference_delta)
            ax.hlines(
                reference_point.mean_interference_delta,
                xmin=x_min,
                xmax=ref_x,
                colors=guide_color,
                linestyles="--",
                linewidth=1.0,
                alpha=guide_alpha,
                zorder=1,
            )
        if guide_values:
            # Clean x-axis intercept annotation.
            ax.annotate(
                f"λ={ref_x:.3f}",
                xy=(ref_x, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -18),
                textcoords="offset points",
                color=guide_color,
                fontsize=8,
                ha="center",
                va="top",
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
            )

            # Clean y-axis intercept annotations with overlap-aware vertical nudges.
            y_labels = [
                ("min", reference_point.min_interference_delta),
                ("mean", reference_point.mean_interference_delta),
            ]
            y_labels = [(name, value) for name, value in y_labels if value is not None]
            y_labels = sorted(y_labels, key=lambda item: item[1])
            for idx, (name, y_value) in enumerate(y_labels):
                y_offset = 0
                if idx > 0 and abs(y_value - y_labels[idx - 1][1]) < 0.02:
                    y_offset = 10
                if name == "min":
                    y_offset -= 8
                ax.annotate(
                    f"{name}={y_value:.3f}",
                    xy=(0, y_value),
                    xycoords=("axes fraction", "data"),
                    xytext=(-8, y_offset),
                    textcoords="offset points",
                    color=guide_color,
                    fontsize=8,
                    ha="right",
                    va="center",
                    clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
                )
    tick_step = 0.1
    tick_min = (int(x_min / tick_step)) * tick_step
    if tick_min < x_min:
        tick_min += tick_step
    tick_max = (int(x_max / tick_step)) * tick_step
    if tick_max < x_max:
        tick_max += tick_step
    xticks = []
    value = tick_min
    while value <= tick_max + 1e-9:
        xticks.append(round(value, 10))
        value += tick_step
    ax.set_xticks(xticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))

    fig.suptitle(title, y=0.98)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot(
    points: list[SweepPoint], *, title: str, output_path: Path, backend: str, reference_lambda: float | None = None
) -> None:
    if backend == "auto":
        try:
            _plot_with_matplotlib(points, title=title, output_path=output_path, reference_lambda=reference_lambda)
            return
        except RuntimeError:
            _plot_with_gnuplot(points, title=title, output_path=output_path, reference_lambda=reference_lambda)
            return
    if backend == "matplotlib":
        _plot_with_matplotlib(points, title=title, output_path=output_path, reference_lambda=reference_lambda)
        return
    if backend == "gnuplot":
        _plot_with_gnuplot(points, title=title, output_path=output_path, reference_lambda=reference_lambda)
        return
    raise ValueError(f"Unknown backend: {backend}")


def shutil_which(cmd: str) -> str | None:
    try:
        import shutil

        return shutil.which(cmd)
    except Exception:
        return None


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep interference delta vs lambda.")
    parser.add_argument(
        "--sweeps-dir",
        type=Path,
        default=Path("artifacts/merged/weighted_delta/emotion_speaker_ver/sweeps"),
        help="Directory containing sweep_*.json files.",
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="Explicit sweep JSON path (overrides --sweeps-dir latest).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/plots/sweeps"),
        help="Directory to write output plot(s).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "gnuplot", "matplotlib"],
        default="matplotlib",
        help="Plotting backend to use (auto prefers matplotlib if available).",
    )
    parser.add_argument(
        "--reference-lambda",
        type=float,
        default=None,
        help="Optional lambda value for red dashed guide lines (nearest sweep point is used).",
    )
    args = parser.parse_args()

    sweep_path = args.sweep if args.sweep is not None else find_latest_sweep(args.sweeps_dir)
    sweep = _load_json(sweep_path)

    adapters = sweep.get("adapters") or []
    adapters_slug = "_".join(str(a) for a in adapters) if adapters else sweep_path.parent.parent.name
    adapters_display = " + ".join(str(a) for a in adapters) if adapters else adapters_slug.replace("_", " ")
    sweep_ts = _parse_sweep_name_timestamp(sweep_path)
    _ = sweep_ts  # retained for potential future use
    title = f"Interference delta vs λ ({adapters_display})"

    points = extract_points(sweep)
    if not points:
        raise ValueError(f"No sweep points found in: {sweep_path}")
    reference_point = _pick_reference_point(points, args.reference_lambda)

    output_path = args.output_dir / f"interference_vs_lambda_{adapters_slug}_{sweep_path.stem}.png"
    plot(
        points,
        title=title,
        output_path=output_path,
        backend=args.backend,
        reference_lambda=args.reference_lambda,
    )

    print(f"Loaded sweep: {sweep_path}")
    print(f"Wrote plot : {output_path}")
    if reference_point is not None:
        print(
            "Guide λ  : "
            f"requested={_format_float(args.reference_lambda) if args.reference_lambda is not None else 'best-min'} "
            f"used={reference_point.lambda_value:.6f}"
        )
    print("")
    print("lambda\tmin_interference_delta\tmean_interference_delta")
    for p in points:
        print(
            f"{p.lambda_value:g}\t"
            f"{_format_float(p.min_interference_delta)}\t"
            f"{_format_float(p.mean_interference_delta)}"
        )


if __name__ == "__main__":
    main()
