"""Rendering of results as terminal ASCII and as publication figures.

The ASCII renderer exists so that CI logs and terminal sessions are
self-contained -- a reviewer should be able to see the shape of a t-trace
without opening a PNG. The matplotlib renderer produces the figures used in
docs/.
"""

from __future__ import annotations

import os

import numpy as np

BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 78) -> str:
    """Downsample `values` to `width` columns and render as block characters.

    Downsampling takes the maximum of each bucket, not the mean: a leak is a
    narrow spike, and averaging would hide exactly what the plot is for.
    """
    v = np.abs(np.asarray(values, dtype=np.float64))
    v = np.nan_to_num(v, posinf=np.nanmax(v[np.isfinite(v)], initial=1.0) * 2)
    if v.size == 0:
        return ""
    buckets = np.array_split(v, min(width, v.size))
    peaks = np.array([b.max() if b.size else 0.0 for b in buckets])
    top = peaks.max() or 1.0
    idx = np.clip((peaks / top * (len(BLOCKS) - 1)).round().astype(int), 0, len(BLOCKS) - 1)
    return "".join(BLOCKS[i] for i in idx)


def t_trace_ascii(t, threshold: float = 4.5, width: int = 78) -> str:
    """Render a t-trace with its threshold marked."""
    t = np.asarray(t, dtype=np.float64)
    finite = t[np.isfinite(t)]
    peak = float(np.abs(finite).max()) if finite.size else float("inf")
    over = int(np.sum(np.abs(t) > threshold))
    bar = sparkline(t, width)
    marks = "".join(
        "^" if b.size and np.abs(b).max() > threshold else " "
        for b in np.array_split(np.asarray(t), min(width, max(t.size, 1)))
    )
    return (
        f"  |t| over {t.size} samples   peak={peak:.2f}   threshold={threshold}\n"
        f"  {bar}\n  {marks}\n"
        f"  {over} sample(s) above threshold (marked ^)"
    )


def correlation_ascii(result, true_key: int | None = None, width: int = 78) -> str:
    """Render peak correlation per key hypothesis, 256 candidates wide."""
    peaks = np.max(result.correlations, axis=1)
    bar = sparkline(peaks, width)
    lines = [f"  peak rho per key candidate (0x00 .. 0xff), best = {result.best:#04x}",
             f"  {bar}"]
    if true_key is not None:
        pos = int(round(true_key / 255 * (width - 1)))
        lines.append("  " + " " * pos + "^ true key")
    return "\n".join(lines)


def save_figures(outdir: str, *, tvla_results=None, cpa_result=None,
                 true_key: int | None = None) -> list[str]:
    """Write PNG figures. Returns the list of paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    written: list[str] = []

    if tvla_results:
        n = len(tvla_results)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=False)
        axes = np.atleast_1d(axes)
        for ax, (name, res) in zip(axes, tvla_results.items(), strict=False):
            t = np.nan_to_num(res.t, posinf=200, neginf=-200)
            ax.plot(t, linewidth=0.7)
            ax.axhline(res.threshold, color="crimson", linestyle="--", linewidth=0.8)
            ax.axhline(-res.threshold, color="crimson", linestyle="--", linewidth=0.8)
            ax.set_title(f"{name}  (max |t| = {res.peak_t:.1f})", fontsize=9)
            ax.set_ylabel("t", fontsize=8)
        axes[-1].set_xlabel("cycle")
        fig.tight_layout()
        path = os.path.join(outdir, "tvla.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    if cpa_result is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        corr = cpa_result.correlations
        for k in range(corr.shape[0]):
            if true_key is not None and k == true_key:
                continue
            ax1.plot(corr[k], color="0.75", linewidth=0.4)
        if true_key is not None:
            ax1.plot(corr[true_key], color="crimson", linewidth=1.4,
                     label=f"true key {true_key:#04x}")
            ax1.legend(fontsize=8)
        ax1.set_title("Correlation per sample, all 256 candidates", fontsize=9)
        ax1.set_xlabel("sample within point of interest")
        ax1.set_ylabel("rho")

        peaks = np.max(corr, axis=1)
        ax2.bar(range(len(peaks)), peaks, width=1.0, color="0.6")
        if true_key is not None:
            ax2.bar([true_key], [peaks[true_key]], width=2.0, color="crimson")
        ax2.set_title("Peak correlation per key candidate", fontsize=9)
        ax2.set_xlabel("key hypothesis")
        ax2.set_ylabel("peak rho")
        fig.tight_layout()
        path = os.path.join(outdir, "cpa.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    return written
