import csv
import math
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from scipy.signal import savgol_filter

# -----------------
# Publication-style defaults
# -----------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

_DISTANCE_COLORS = {
    1: "#1b9e77",
    3: "#d95f02",
    5: "#7570b3",
    7: "#e7298a",
}
_DISTANCE_MARKERS = {1: "o", 3: "s", 5: "D", 7: "^"}

_UNIT_INTERVAL_METRICS = {
    "success_prob_within_cap",
    "erasure_rate",
    "avg_fidelity_injected_given_success",
    "avg_fidelity_logical_given_success",
}


def _is_unit_interval_metric(metric: str) -> bool:
    if metric in _UNIT_INTERVAL_METRICS:
        return True
    if metric.startswith("avg_fidelity"):
        return True
    if metric.startswith("p_"):
        return True
    if metric.endswith("_prob") or metric.endswith("_probability"):
        return True
    if metric.endswith("_rate"):
        return True
    return False

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
    y_limits: Tuple[float, float] | None = None,
    auto_zoom_y: bool = True,
):
    """
    Plot metric vs squeezing for each loss and code distance.
    Raw Monte Carlo points are shown as markers.
    Optional Savitzky–Golay smoothing is used for visual guidance only.
    """

    loss_values = sorted(set(r["loss_base"] for r in rows))
    distances = sorted(set(int(r["distance"]) for r in rows))

    n_loss = len(loss_values)
    ncols = min(3, n_loss) if n_loss > 0 else 1
    nrows = int(math.ceil(n_loss / ncols)) if n_loss > 0 else 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.0 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.array(axes, dtype=object).reshape(-1)

    def _savgol_safe(y: np.ndarray, w: int, p: int) -> np.ndarray | None:
        n = len(y)
        if n < 3:
            return None
        w = min(w, n if n % 2 == 1 else n - 1)
        if w < 3:
            return None
        if w <= p:
            p = max(1, w - 1)
        if w % 2 == 0:
            w -= 1
        if w < 3 or w <= p:
            return None
        return savgol_filter(y, window_length=w, polyorder=p, mode="interp")

    # Compute global y-limits if requested
    y_min = None
    y_max = None
    if y_limits is None and auto_zoom_y:
        all_vals = [float(r[metric]) for r in rows if metric in r]
        if all_vals:
            y_min = float(min(all_vals))
            y_max = float(max(all_vals))
            if y_min == y_max:
                pad = max(1e-3, 0.02 * abs(y_min))
            else:
                pad = max(1e-3, 0.05 * (y_max - y_min))
            y_min -= pad
            y_max += pad
            if _is_unit_interval_metric(metric):
                y_min = max(0.0, y_min)
                y_max = min(1.0, y_max)

    for ax_i, ax in enumerate(axes):
        if ax_i >= n_loss:
            ax.set_visible(False)
            continue
        loss_base = loss_values[ax_i]
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

            color = _DISTANCE_COLORS.get(d, None)
            marker = _DISTANCE_MARKERS.get(d, "o")
            if color is None:
                color = plt.cm.tab10((d * 3) % 10)

            # raw data (transparent markers)
            ax.scatter(xs, ys, s=14, alpha=0.35, color=color, edgecolors="none")

            # smoothed trend (guide to eye)
            ys_smooth = _savgol_safe(ys, window, polyorder) if smooth else None
            if ys_smooth is not None:
                ax.plot(
                    xs,
                    ys_smooth,
                    linewidth=1.8,
                    color=color,
                    marker=marker,
                    markersize=3,
                    markevery=2,
                    label=f"d={d}",
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    linewidth=1.4,
                    color=color,
                    marker=marker,
                    markersize=3,
                    markevery=2,
                    label=f"d={d}",
                )

        ax.set_title(f"loss = {loss_base:.3f}")
        ax.set_xlabel("Squeezing (dB)")
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

        if y_limits is not None:
            ax.set_ylim(y_limits[0], y_limits[1])
        elif auto_zoom_y and y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        elif _is_unit_interval_metric(metric):
            ax.set_ylim(0.0, 1.0)

    if n_loss > 0:
        axes[0].set_ylabel(ylabel)

    # Global legend
    handles, labels = [], []
    for ax in axes:
        if ax.get_visible():
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            title="Code distance",
        )

    fig.suptitle(title, y=1.04)
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
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

    # Precompute gradients for consistent color scaling
    grids = []
    for d in distances:
        losses, squeezes, M = _grid_metric(rows, metric=metric, distance=d)
        dM_dloss, dM_ds = _finite_diff_gradients(losses, squeezes, M)
        grids.append((d, losses, squeezes, dM_dloss, dM_ds))

    def _sym_max(arrs: List[np.ndarray]) -> float:
        vals = []
        for a in arrs:
            if a.size == 0:
                continue
            finite = a[np.isfinite(a)]
            if finite.size:
                vals.append(np.max(np.abs(finite)))
        if not vals:
            return 1.0
        return float(max(vals))

    def _robust_sym_max(arr: np.ndarray, q: float = 90.0) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 1.0
        abs_vals = np.abs(finite)
        vmax = float(np.percentile(abs_vals, q))
        if vmax <= 0:
            vmax = float(np.max(abs_vals))
        if vmax <= 0:
            vmax = 1.0
        return vmax

    vmax_loss = _sym_max([g[3] for g in grids])
    vmax_sq = _sym_max([g[4] for g in grids])

    for d, losses, squeezes, dM_dloss, dM_ds in grids:
        # Common plotting extents
        extent = [squeezes.min(), squeezes.max(), losses.min(), losses.max()]

        # (1) d/dloss
        plt.figure(figsize=(5.4, 4.0))
        norm = TwoSlopeNorm(vmin=-vmax_loss, vcenter=0.0, vmax=vmax_loss)
        im = plt.imshow(
            dM_dloss,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            norm=norm,
        )
        cbar = plt.colorbar(im)
        cbar.set_label(f"∂({metric})/∂loss")
        cbar.ax.tick_params(labelsize=8)
        plt.xlabel("Squeezing (dB)")
        plt.ylabel("loss_base")
        plt.title(f"Sensitivity to loss (d={d})")
        plt.tight_layout()
        plt.savefig(f"{outfile_prefix}_d{d}_dF_dloss.png", dpi=600, bbox_inches="tight")
        plt.close()

        # (2) d/dsqueeze (use local, robust scaling for better contrast)
        plt.figure(figsize=(5.4, 4.0))
        vmax_local = _robust_sym_max(dM_ds, q=90.0)
        norm = TwoSlopeNorm(vmin=-vmax_local, vcenter=0.0, vmax=vmax_local)
        im = plt.imshow(
            dM_ds,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            norm=norm,
        )
        cbar = plt.colorbar(im)
        cbar.set_label(f"∂({metric})/∂squeezing")
        cbar.ax.tick_params(labelsize=8)
        plt.xlabel("Squeezing (dB)")
        plt.ylabel("loss_base")
        plt.title(f"Sensitivity to squeezing (d={d})")
        plt.tight_layout()
        plt.savefig(f"{outfile_prefix}_d{d}_dF_dsqueezing.png", dpi=600, bbox_inches="tight")
        plt.close()

def plot_phase_boundary(
    rows: List[Dict[str, float]],
    metric_success: str = "success_prob_within_cap",
    metric_fidelity: str = "avg_fidelity_logical_given_success",
    success_thresh: float = 0.95,
    fidelity_thresh: float = 0.79,
    outfile: str = "fig_phase_boundary.png",
    y_limits: Tuple[float, float] | None = None,
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

    plt.figure(figsize=(6.5, 4.5))

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
            color = _DISTANCE_COLORS.get(d, None)
            marker = _DISTANCE_MARKERS.get(d, "o")
            if color is None:
                color = plt.cm.tab10((d * 3) % 10)
            plt.plot(
                boundary_l,
                boundary_s,
                marker=marker,
                linewidth=2,
                markersize=4,
                color=color,
                label=f"d={d}",
            )

    plt.xlabel("loss_base")
    plt.ylabel("Minimal squeezing (dB)")
    if y_limits is not None:
        plt.ylim(y_limits[0], y_limits[1])
    plt.title(
        f"Phase boundary: success ≥ {success_thresh:.2f} and fidelity ≥ {fidelity_thresh:.2f}"
    )
    plt.grid(alpha=0.2, linewidth=0.6)
    plt.legend(frameon=False, title="Code distance")
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close()


def plot_rus_t_injection_circuit(
    outfile: str = "fig_rus_t_injection_circuit.png",
) -> None:
    """
    Draw a clean, publication-style schematic of the logical RUS T-injection circuit.
    """
    # Disabled per request: do not generate this figure.
    return
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    ax.axis("off")

    # Coordinates
    y_data = 1.0
    y_anc = 0.0
    x_left = 0.1
    x_right = 0.9

    # Wires
    ax.plot([x_left, x_right], [y_data, y_data], color="black", linewidth=1.4)
    ax.plot([x_left, x_right], [y_anc, y_anc], color="black", linewidth=1.4)

    # Labels
    ax.text(x_left - 0.04, y_data, r"data $|\psi\rangle$", va="center", ha="right")
    ax.text(x_left - 0.04, y_anc, r"anc $|A\rangle$", va="center", ha="right")

    # CNOT (control on data, target on ancilla)
    x_cnot = 0.45
    ax.add_patch(Circle((x_cnot, y_data), 0.012, color="black"))
    ax.plot([x_cnot, x_cnot], [y_anc, y_data], color="black", linewidth=1.2)
    ax.add_patch(Circle((x_cnot, y_anc), 0.03, fill=False, linewidth=1.2, color="black"))
    ax.plot([x_cnot - 0.03, x_cnot + 0.03], [y_anc, y_anc], color="black", linewidth=1.2)
    ax.plot([x_cnot, x_cnot], [y_anc - 0.03, y_anc + 0.03], color="black", linewidth=1.2)

    # Measurement in X basis on ancilla
    x_meas = 0.65
    meas_box = Rectangle((x_meas - 0.045, y_anc - 0.06), 0.09, 0.12, fill=False, linewidth=1.2)
    ax.add_patch(meas_box)
    ax.text(x_meas, y_anc, r"$M_X$", ha="center", va="center")

    # Feedforward S / S^\dagger on data
    x_ff = 0.75
    ff_box = Rectangle((x_ff - 0.06, y_data - 0.06), 0.12, 0.12, fill=False, linewidth=1.2)
    ax.add_patch(ff_box)
    ax.text(x_ff, y_data, r"$S/S^\dagger$", ha="center", va="center")

    # Classical control arrow from measurement to feedforward
    arrow = FancyArrowPatch(
        (x_meas + 0.045, y_anc + 0.02),
        (x_ff - 0.06, y_data - 0.02),
        arrowstyle="->",
        mutation_scale=10,
        linewidth=1.0,
        linestyle="--",
        color="black",
    )
    ax.add_patch(arrow)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.25, 1.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close(fig)
