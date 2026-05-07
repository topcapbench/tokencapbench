from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _savefig(fig, output_path: Path) -> None:
    suffix = output_path.suffix.lower().lstrip(".") or "svg"
    fig.savefig(output_path, format=suffix)


def _write_svg_scatter(points: list[tuple[float, float]], output_path: str | Path, title: str, xlabel: str, ylabel: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 420
    pad = 60
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x = min_y = 0.0
        max_x = max_y = 1.0
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0

    def sx(x: float) -> float:
        return pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - min_y) / (max_y - min_y) * (height - 2 * pad)

    circles = "\n".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" />' for x, y in points)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>
  <text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="13">{xlabel}</text>
  <text x="20" y="{height/2}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 20 {height/2})">{ylabel}</text>
  <g fill="black" fill-opacity="0.75">{circles}</g>
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def reliability_plot(probabilities: list[float], outcomes: list[bool], output_path: str | Path, n_bins: int = 10) -> None:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs: list[float] = []
    ys: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if np.any(mask):
            xs.append(float(np.mean(p[mask])))
            ys.append(float(np.mean(y[mask])))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([0, 1], [0, 1], color="#6b7280", linewidth=1.2, linestyle="--", label="Perfect calibration")
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=1.8, color="#2563eb", label="Observed")
    ax.set_title("Budgeted Success Calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed success rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def scatter_predicted_observed(predicted: list[float], observed: list[float], output_path: str | Path) -> None:
    points = [(float(np.log10(p)), float(np.log10(o))) for p, o in zip(predicted, observed) if p > 0 and o > 0]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, color="#111827", alpha=0.75)
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], color="#6b7280", linewidth=1.2, linestyle="--")
    ax.set_title("Predicted vs. Observed TokenCapBench")
    ax.set_xlabel("log10 predicted tokens")
    ax.set_ylabel("log10 observed tokens")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def success_rate_by_budget_plot(
    series: dict[str, dict[int, float]], output_path: str | Path, title: str = "Pilot Budget-Success Curves"
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, rates in sorted(series.items()):
        if not rates:
            continue
        budgets = sorted(rates)
        ax.plot(budgets, [rates[b] for b in budgets], marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Generated-token budget")
    ax.set_ylabel("Verified success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    if series:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def regret_curve_plot(curves: dict[str, list[tuple[float, float]]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, points in sorted(curves.items()):
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
    ax.set_title("Budget-Selection Regret")
    ax.set_xlabel("Token cost lambda")
    ax.set_ylabel("Mean regret")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.grid(True, alpha=0.25)
    if curves:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)
