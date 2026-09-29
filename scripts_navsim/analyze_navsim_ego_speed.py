"""Analyze ego speed distribution of NAVSIM logs and plot histograms.

Ego speed is derived from each frame's ``ego_dynamic_state = [vx, vy, ax, ay]``
(local ego frame), speed = sqrt(vx^2 + vy^2) in m/s.

Usage:
    ./venv/bin/python scripts_clapad/analyze_navsim_ego_speed.py \
        --splits test trainval --out outputs/navsim_ego_speed
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Defaults to <repo>/3rd_party/navsim_workspace/dataset/navsim_logs, or
# $OPENSCENE_DATA_ROOT/navsim_logs if that environment variable is set (see
# scripts_navsim/navsim_env.sh).
_REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.environ.get("OPENSCENE_DATA_ROOT", _REPO_ROOT / "3rd_party/navsim_workspace/dataset")
) / "navsim_logs"


def collect_speeds(split_dir: Path) -> np.ndarray:
    speeds = []
    for pkl in sorted(split_dir.glob("*.pkl")):
        with open(pkl, "rb") as f:
            frames = pickle.load(f)
        for fr in frames:
            vx, vy = fr["ego_dynamic_state"][0], fr["ego_dynamic_state"][1]
            speeds.append((vx * vx + vy * vy) ** 0.5)
    return np.asarray(speeds, dtype=np.float64)


def summarize(name: str, s: np.ndarray) -> str:
    q = np.percentile(s, [0, 25, 50, 75, 90, 95, 99, 100])
    lines = [
        f"[{name}] n={s.size}  mean={s.mean():.3f}  std={s.std():.3f} m/s",
        f"    min={q[0]:.2f}  p25={q[1]:.2f}  median={q[2]:.2f}  "
        f"p75={q[3]:.2f}  p90={q[4]:.2f}  p95={q[5]:.2f}  "
        f"p99={q[6]:.2f}  max={q[7]:.2f} m/s",
        f"    stationary (<0.2 m/s): {100 * np.mean(s < 0.2):.1f}%   "
        f"mean km/h={s.mean() * 3.6:.1f}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--out", type=Path, default=Path("outputs/navsim_ego_speed"))
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--max-speed", type=float, default=20.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = {}
    for split in args.splits:
        split_dir = args.root / split
        if not split_dir.is_dir():
            print(f"!! skip missing split: {split_dir}")
            continue
        s = collect_speeds(split_dir)
        data[split] = s
        print(summarize(split, s))
        np.save(args.out / f"ego_speed_{split}.npy", s)

    if not data:
        return

    bins = np.linspace(0, args.max_speed, args.bins + 1)

    # Individual histograms
    for split, s in data.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        counts, _, _ = ax.hist(np.clip(s, 0, args.max_speed), bins=bins,
                               color="#3b7dd8", edgecolor="white", linewidth=0.3)
        ax.axvline(s.mean(), color="#d84b3b", linestyle="--",
                   label=f"mean {s.mean():.2f} m/s")
        ax.axvline(np.median(s), color="#2ca02c", linestyle="--",
                   label=f"median {np.median(s):.2f} m/s")
        ax.set_xlabel("Ego speed (m/s)")
        ax.set_ylabel("Frame count")
        ax.set_title(f"NAVSIM ego speed distribution — {split} (n={s.size})")

        # Rescale y-axis to ignore the (dominant) zero/stationary bin so the
        # rest of the distribution is legible; annotate its true count instead.
        zero_count = int(counts[0])
        if counts.size > 1:
            ymax = counts[1:].max() * 1.15
            ax.set_ylim(0, ymax)
            ax.annotate(
                f"zero bin (0–{bins[1]:.2f} m/s):\n{zero_count:,} frames "
                f"({100 * zero_count / s.size:.1f}%)  ↑ off-scale",
                xy=(bins[0], ymax), xytext=(0.32, 0.9),
                textcoords="axes fraction", ha="left", va="top",
                fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#888888"),
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff4e6",
                          ec="#e0a44b", alpha=0.9),
            )
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = args.out / f"ego_speed_hist_{split}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print(f"saved {p}")

    # Combined normalized overlay
    if len(data) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for split, s in data.items():
            ax.hist(np.clip(s, 0, args.max_speed), bins=bins, density=True,
                    histtype="step", linewidth=1.8,
                    label=f"{split} (mean {s.mean():.2f})")
        ax.set_xlabel("Ego speed (m/s)")
        ax.set_ylabel("Density")
        ax.set_title("NAVSIM ego speed distribution (normalized)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = args.out / "ego_speed_hist_combined.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print(f"saved {p}")


if __name__ == "__main__":
    main()
