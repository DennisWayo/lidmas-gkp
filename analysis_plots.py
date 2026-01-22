import csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#------------------
# Analysis
#------------------

def write_csv(path: str, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def plot_metric_vs_squeezing_by_loss(
    rows,
    metric: str,
    ylabel: str,
    title: str,
    outfile: str,
    smooth: bool = True,
    window: int = 7,
    polyorder: int = 2,
):
    """
    Plot metric vs squeezing for each loss and code distance.
    Raw Monte Carlo points are shown as markers.
    Optional Savitzky–Golay smoothing is used for visual guidance only.
    """

    loss_values = sorted(set(r["loss_base"] for r in rows))
    distances = sorted(set(int(r["distance"]) for r in rows))

    fig, axes = plt.subplots(
        1, len(loss_values),
        figsize=(5 * len(loss_values), 4),
        sharey=True
    )

    if len(loss_values) == 1:
        axes = [axes]

    for ax, loss_base in zip(axes, loss_values):
        for d in distances:
            xs = np.array([
                r["squeezing_db"] for r in rows
                if r["loss_base"] == loss_base and int(r["distance"]) == d
            ])
            ys = np.array([
                r[metric] for r in rows
                if r["loss_base"] == loss_base and int(r["distance"]) == d
            ])

            if len(xs) < 3:
                continue

            # sort by squeezing
            order = np.argsort(xs)
            xs = xs[order]
            ys = ys[order]

            # raw data (transparent markers)
            ax.scatter(xs, ys, s=18, alpha=0.6)

            # smoothed trend (guide to eye)
            if smooth and len(xs) >= window:
                ys_smooth = savgol_filter(
                    ys,
                    window_length=window,
                    polyorder=polyorder,
                    mode="interp"
                )
                ax.plot(xs, ys_smooth, linewidth=2, label=f"d={d}")
            else:
                ax.plot(xs, ys, linewidth=1.5, label=f"d={d}")

        ax.set_title(f"loss = {loss_base:.3f}")
        ax.set_xlabel("Squeezing proxy (dB)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(title="Code distance", frameon=False)

    fig.suptitle(title, y=1.05)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _unique_sorted(rows: List[Dict[str, float]], key: str) -> List[float]:
    return sorted(set(float(r[key]) for r in rows))


def _grid_metric(
    rows: List[Dict[str, float]],
    metric: str,
    distance: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2D grid of metric values over (loss_base, squeezing_db) for a fixed distance.

    Returns:
      L (n_loss,), S (n_sq,), M (n_loss, n_sq)
    """
    losses = _unique_sorted(rows, "loss_base")
    squeezes = _unique_sorted(rows, "squeezing_db")

    # map (loss, squeeze) -> metric
    lookup = {}
    for r in rows:
        if int(r["distance"]) != int(distance):
            continue
        key = (float(r["loss_base"]), float(r["squeezing_db"]))
        lookup[key] = float(r[metric])

    M = np.full((len(losses), len(squeezes)), np.nan, dtype=float)
    for i, L in enumerate(losses):
        for j, S in enumerate(squeezes):
            M[i, j] = lookup.get((L, S), np.nan)

    return np.array(losses), np.array(squeezes), M


def _finite_diff_gradients(losses: np.ndarray, squeezes: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Central finite differences where possible; one-sided at edges.
    Returns dM/dloss and dM/dsqueeze, same shape as M.

    """
    dM_dloss = np.full_like(M, np.nan)
    dM_ds    = np.full_like(M, np.nan)

    # d/dloss (axis 0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            if 0 < i < M.shape[0] - 1:
                if not (np.isnan(M[i-1, j]) or np.isnan(M[i+1, j])):
                    dL = losses[i+1] - losses[i-1]
                    dM_dloss[i, j] = (M[i+1, j] - M[i-1, j]) / dL
            elif i == 0 and M.shape[0] > 1:
                if not np.isnan(M[i+1, j]):
                    dL = losses[i+1] - losses[i]
                    dM_dloss[i, j] = (M[i+1, j] - M[i, j]) / dL
            elif i == M.shape[0] - 1 and M.shape[0] > 1:
                if not np.isnan(M[i, j]) and not np.isnan(M[i-1, j]):
                    dL = losses[i] - losses[i-1]
                    dM_dloss[i, j] = (M[i, j] - M[i-1, j]) / dL

    # d/dsqueeze (axis 1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            if 0 < j < M.shape[1] - 1:
                if not (np.isnan(M[i, j-1]) or np.isnan(M[i, j+1])):
                    dS = squeezes[j+1] - squeezes[j-1]
                    dM_ds[i, j] = (M[i, j+1] - M[i, j-1]) / dS
            elif j == 0 and M.shape[1] > 1:
                if not np.isnan(M[i, j+1]):
                    dS = squeezes[j+1] - squeezes[j]
                    dM_ds[i, j] = (M[i, j+1] - M[i, j]) / dS
            elif j == M.shape[1] - 1 and M.shape[1] > 1:
                if not np.isnan(M[i, j]) and not np.isnan(M[i, j-1]):
                    dS = squeezes[j] - squeezes[j-1]
                    dM_ds[i, j] = (M[i, j] - M[i, j-1]) / dS

    return dM_dloss, dM_ds


def plot_sensitivity_heatmaps(
    rows: List[Dict[str, float]],
    metric: str = "avg_fidelity_logical_given_success",
    outfile_prefix: str = "fig_sensitivity",
) -> None:
    """
    Produces, for each distance d:
      - heatmap of d(metric)/d(loss_base)
      - heatmap of d(metric)/d(squeezing_db)
    """
    distances = sorted(set(int(r["distance"]) for r in rows))

    for d in distances:
        losses, squeezes, M = _grid_metric(rows, metric=metric, distance=d)
        dM_dloss, dM_ds = _finite_diff_gradients(losses, squeezes, M)

        # Common plotting extents
        extent = [squeezes.min(), squeezes.max(), losses.min(), losses.max()]

        # (1) d/dloss
        plt.figure(figsize=(6, 4.5))
        plt.imshow(dM_dloss, origin="lower", aspect="auto", extent=extent)
        plt.colorbar(label=f"∂({metric})/∂loss")
        plt.xlabel("Squeezing proxy (dB)")
        plt.ylabel("loss_base")
        plt.title(f"Sensitivity to loss (distance d={d})")
        plt.tight_layout()
        plt.savefig(f"{outfile_prefix}_d{d}_dF_dloss.png", dpi=300, bbox_inches="tight")
        plt.close()

        # (2) d/dsqueeze
        plt.figure(figsize=(6, 4.5))
        plt.imshow(dM_ds, origin="lower", aspect="auto", extent=extent)
        plt.colorbar(label=f"∂({metric})/∂squeezing")
        plt.xlabel("Squeezing proxy (dB)")
        plt.ylabel("loss_base")
        plt.title(f"Sensitivity to squeezing (distance d={d})")
        plt.tight_layout()
        plt.savefig(f"{outfile_prefix}_d{d}_dF_dsqueezing.png", dpi=300, bbox_inches="tight")
        plt.close()

def plot_phase_boundary(
    rows: List[Dict[str, float]],
    metric_success: str = "success_prob_within_cap",
    metric_fidelity: str = "avg_fidelity_logical_given_success",
    success_thresh: float = 0.95,
    fidelity_thresh: float = 0.79,
    outfile: str = "fig_phase_boundary.png",
) -> None:
    """
    Phase boundary: for each code distance, mark the (loss, squeezing) region
    where BOTH:
      success_prob_within_cap >= success_thresh
      avg_fidelity_logical_given_success >= fidelity_thresh
    and plot an approximate boundary curve: minimal squeezing required vs loss.
    """
    losses = _unique_sorted(rows, "loss_base")
    squeezes = _unique_sorted(rows, "squeezing_db")
    distances = sorted(set(int(r["distance"]) for r in rows))

    # lookup for fast access
    lookup = {}
    for r in rows:
        key = (float(r["loss_base"]), float(r["squeezing_db"]), int(r["distance"]))
        lookup[key] = (float(r[metric_success]), float(r[metric_fidelity]))

    plt.figure(figsize=(7, 5))

    for d in distances:
        boundary_s = []
        boundary_l = []

        for L in losses:
            # find smallest squeezing that satisfies both conditions
            s_ok = None
            for S in squeezes:
                vals = lookup.get((L, S, d), None)
                if vals is None:
                    continue
                succ, fid = vals
                if (succ >= success_thresh) and (fid >= fidelity_thresh):
                    s_ok = S
                    break
            if s_ok is not None:
                boundary_l.append(L)
                boundary_s.append(s_ok)

        if boundary_l:
            plt.plot(boundary_l, boundary_s, marker="o", linewidth=2, label=f"d={d}")

    plt.xlabel("loss_base")
    plt.ylabel("Minimal squeezing (dB) to meet thresholds")
    plt.title(
        f"Phase boundary: success ≥ {success_thresh:.2f} and fidelity ≥ {fidelity_thresh:.2f}"
    )
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, title="Code distance")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()