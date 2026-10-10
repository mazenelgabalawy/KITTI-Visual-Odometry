"""Trajectory and error plots (xz plane), written with matplotlib's Agg backend.

Importing evo.tools.plot calls ``mpl.use(SETTINGS.plot_backend)``, which
defaults to an interactive Qt backend. Without a Qt binding installed (CI, the
plain `eval` extra) that import raises ImportError regardless of MPLBACKEND,
and with one it overrides an earlier ``matplotlib.use("Agg")``. evo's
in-process plot backend setting is therefore switched to Agg *before* the
import; this affects only the current process, not the user's evo config file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kitti_slam.evaluation.metrics import MetricResult, poses_to_trajectory


def _plot_module():
    import matplotlib
    from evo.tools.settings import SETTINGS

    SETTINGS.plot_backend = "Agg"
    from evo.tools import plot

    matplotlib.use("Agg")  # in case evo.tools.plot was imported earlier with another backend
    import matplotlib.pyplot as plt

    return plot, plt


def save_trajectory_plot(gt: np.ndarray, est: np.ndarray, out_path: str | Path) -> None:
    """Reference vs. estimate in the xz plane (cf. `evo_traj kitti --ref gt est`)."""
    plot, plt = _plot_module()
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot.PlotMode.xz)
    plot.traj(ax, plot.PlotMode.xz, poses_to_trajectory(gt), style="--", color="gray", label="reference")
    plot.traj(ax, plot.PlotMode.xz, poses_to_trajectory(est), style="-", color="blue", label="estimate")
    ax.legend(frameon=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_error_plots(gt: np.ndarray, est: np.ndarray, result: MetricResult, out_prefix: str | Path) -> None:
    """`<prefix>_raw.png` (error per pose or pair, labeled with its SI unit) and
    `<prefix>_map.png` (estimate colored by error). Pairwise errors are drawn
    at each pair's end pose, as evo does. No-op for an unavailable metric."""
    if not result.available or result.errors is None:
        return
    plot, plt = _plot_module()
    errors = result.errors

    fig = plt.figure()
    plot.error_array(
        fig.gca(),
        errors,
        statistics=dict(result.statistics.__dict__),
        name=f"{result.name} ({result.unit})",
        xlabel="pair index" if result.pairs is not None else "pose index",
    )
    fig.savefig(f"{out_prefix}_raw.png")
    plt.close(fig)

    ids = [j for _, j in result.pairs] if result.pairs is not None else list(range(len(errors)))
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot.PlotMode.xz)
    plot.traj(ax, plot.PlotMode.xz, poses_to_trajectory(gt), style="--", color="gray", label="reference")
    plot.traj_colormap(
        ax,
        poses_to_trajectory(np.asarray(est)[ids]),
        errors,
        plot.PlotMode.xz,
        min_map=float(np.min(errors)),
        max_map=float(np.max(errors)),
    )
    ax.set_title(f"{result.name} ({result.unit})")
    fig.savefig(f"{out_prefix}_map.png")
    plt.close(fig)
