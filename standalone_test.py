#!/usr/bin/env python3
"""
TrackDLO Standalone Test
========================
Run the TrackDLO CPD-LLE tracking algorithm on a synthetic point-cloud
sequence **without ROS**.  Requires only numpy, scipy, matplotlib, and
open3d (optional, for 3-D visualisation).

The compiled pybind11 extension ``trackdlo_cpp`` must be on PYTHONPATH.
Run from the repo root:

    python standalone_test.py                      # headless, saves plots
    python standalone_test.py --visualize-3d       # open3d window per frame
    python standalone_test.py --noise 0.005        # add Gaussian noise [m]
    python standalone_test.py --frames 30          # number of timesteps

Background
----------
TrackDLO (Holly Dinkel et al., RA-L 2023) tracks a DLO as a sequence of
3-D nodes by fitting a Coherent Point Drift (CPD) mixture model regularised
with Locally Linear Embedding (LLE) structure weights.  The C++ backend
(trackdlo_cpp) is wrapped via pybind11 — see pybind_ext/trackdlo_bind.cpp.

This script synthesises a rope that droops under simulated gravity and
supplies each frame as a point cloud to the tracker, so the full pipeline
can be exercised without a ROS/camera setup.
"""

import argparse
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── locate and import the pybind extension ────────────────────────────────────
_EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pybind_ext')
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

try:
    import trackdlo_cpp
except ImportError as e:
    sys.exit(
        f"Cannot import trackdlo_cpp: {e}\n"
        f"Make sure the .so in pybind_ext/ matches your Python version "
        f"and that you are in the correct conda environment.\n"
        f"Looked in: {_EXT_DIR}"
    )

# ── optional open3d ───────────────────────────────────────────────────────────
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic rope generator
# ─────────────────────────────────────────────────────────────────────────────

def _catenary(s, sag):
    """Return (x, z) of a catenary with arc-length parameter s ∈ [0, L]."""
    L = s[-1]
    x = s  # x runs from 0 → L (no centring — matches CPD initialisation)
    # approximate catenary: z = -sag * (1 - (2*(x-L/2)/L)^2)  (sags downward)
    z = -sag * (1.0 - (2.0 * (x - L / 2) / L) ** 2)
    return x, z


def make_rope_sequence(n_nodes=40, L=0.8, n_frames=20,
                       max_sag=0.15, noise_std=0.003,
                       pts_per_node=8):
    """
    Synthesise a rope drooping under gravity over `n_frames` timesteps.

    Returns
    -------
    frames : list of (N_pts, 3) float64 arrays  — point clouds (one per frame)
    gt_nodes : list of (n_nodes, 3) float64 arrays — ground-truth node positions
    """
    s = np.linspace(0, L, n_nodes)
    frames, gt_nodes = [], []

    for t in range(n_frames):
        sag = max_sag * (t / max(n_frames - 1, 1))
        x_gt, z_gt = _catenary(s, sag)
        y_gt = np.zeros_like(x_gt)
        nodes_gt = np.column_stack([x_gt, y_gt, z_gt])  # (n_nodes, 3)

        # Dense point cloud: sample points along segments with noise
        pts = []
        for i in range(n_nodes - 1):
            alphas = np.linspace(0, 1, pts_per_node, endpoint=False)
            for a in alphas:
                pt = (1 - a) * nodes_gt[i] + a * nodes_gt[i + 1]
                pt += np.random.randn(3) * noise_std
                pts.append(pt)
        pts = np.array(pts, dtype=np.float64)

        gt_nodes.append(nodes_gt.copy())
        frames.append(pts)

    return frames, gt_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Tracking pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_tracker(frames, gt_nodes, n_nodes=40, L=0.8):
    """
    Run trackdlo_cpp on a list of point-cloud frames.

    Returns
    -------
    results  : list of (n_nodes, 3) tracked node positions
    errors   : list of float — mean Euclidean error vs ground truth [m]
    times_ms : list of float — per-frame tracking time [ms]
    """
    tracker = trackdlo_cpp.Tracker(
        num_of_nodes=n_nodes,
        visibility_threshold=0.01,
        beta=5.0,
        lambda_=1.0,
        alpha=0.0,
        k_vis=0.0,
        mu=0.05,
        max_iter=30,
        tol=1e-4,
        beta_pre_proc=0.5,
        lambda_pre_proc=1.0,
        lle_weight=1.0,
    )

    # ── Initialisation: register first frame ──────────────────────────────────
    pts0 = frames[0]
    Y0, sigma2_0 = trackdlo_cpp.reg(pts0, n_nodes, mu=0.05, max_iter=100)
    Y0 = trackdlo_cpp.sort_pts(Y0)

    tracker.initialize_nodes(Y0)

    # Set up geodesic coordinates from initial node spacing
    seg_dis = np.sqrt(np.sum(np.diff(Y0, axis=0) ** 2, axis=1))
    geo_coord = np.concatenate([[0.0], np.cumsum(seg_dis)]).tolist()
    tracker.initialize_geodesic_coord(geo_coord)
    tracker.set_sigma2(sigma2_0)

    # Projection matrix (identity — no image projection in standalone mode)
    proj = np.hstack([np.eye(3), np.zeros((3, 1))])

    results, errors, times_ms = [], [], []

    # ── Frame loop ────────────────────────────────────────────────────────────
    for frame_idx, (pts, gt) in enumerate(zip(frames, gt_nodes)):
        t0 = time.perf_counter()

        # All nodes visible (no occlusion in synthetic data)
        visible = list(range(n_nodes))

        # Use cpd_lle directly (tracking_step requires image dimensions)
        Y = tracker.get_tracking_result()
        sigma2 = tracker.get_sigma2()
        Y, sigma2, converged = tracker.cpd_lle(
            X=pts, Y=Y, sigma2=sigma2,
            beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05,
            max_iter=30, tol=1e-4, include_lle=True,
        )
        # Write result back
        tracker.initialize_nodes(Y)
        tracker.set_sigma2(sigma2)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Mean per-node error vs ground truth
        err = np.mean(np.linalg.norm(Y - gt, axis=1))

        results.append(Y.copy())
        errors.append(err)
        times_ms.append(elapsed_ms)

        print(
            f"  frame {frame_idx:3d}  |  "
            f"error = {err*1000:6.2f} mm  |  "
            f"time = {elapsed_ms:6.1f} ms  |  "
            f"{'converged' if converged else 'max_iter'}"
        )

    return results, errors, times_ms


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(results, gt_nodes, errors, times_ms, noise_std):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── Panel 1: tracked vs ground-truth shapes (first, mid, last frame) ──
    ax = axes[0]
    palette = {'first': 'steelblue', 'mid': 'mediumseagreen', 'last': 'tomato'}
    idxs = {'first': 0, 'mid': len(results) // 2, 'last': len(results) - 1}
    for label, fi in idxs.items():
        gt = gt_nodes[fi]
        tr = results[fi]
        c  = palette[label]
        ax.plot(gt[:, 0], gt[:, 2], '--', color=c, lw=1.5, alpha=0.6,
                label=f'{label} GT')
        ax.plot(tr[:, 0], tr[:, 2], '-o', color=c, lw=2, ms=3,
                label=f'{label} tracked')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('Tracked vs Ground-Truth Shape\n(3 selected frames)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: per-frame tracking error ──
    ax2 = axes[1]
    frames_axis = np.arange(len(errors))
    ax2.plot(frames_axis, np.array(errors) * 1000, 'b-o', ms=4, lw=2)
    ax2.axhline(np.array(errors).mean() * 1000, color='r', ls='--',
                label=f'mean = {np.array(errors).mean()*1000:.2f} mm')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Mean node error [mm]')
    ax2.set_title(f'Tracking error per frame\n(noise σ = {noise_std*1000:.1f} mm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: per-frame computation time ──
    ax3 = axes[2]
    ax3.bar(frames_axis, times_ms, color='steelblue', alpha=0.7)
    ax3.axhline(np.mean(times_ms), color='r', ls='--',
                label=f'mean = {np.mean(times_ms):.1f} ms')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Time [ms]')
    ax3.set_title('Computation time per frame')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'trackdlo_standalone_results.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f'\nFigure saved to {out}')
    plt.show()


def visualize_3d_frame(pts, gt, tracked, frame_idx):
    """Open3D live visualisation for a single frame."""
    if not HAS_O3D:
        print('[visualize_3d] open3d not installed — skipping.')
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    def _nodes_to_lineset(nodes, color):
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(nodes)
        ls.lines  = o3d.utility.Vector2iVector(
            [[i, i+1] for i in range(len(nodes)-1)])
        ls.paint_uniform_color(color)
        return ls

    gt_ls  = _nodes_to_lineset(gt,      [0.2, 0.8, 0.2])   # green = GT
    tr_ls  = _nodes_to_lineset(tracked, [0.9, 0.3, 0.1])   # red = tracked

    o3d.visualization.draw_geometries(
        [pcd, gt_ls, tr_ls],
        window_name=f'TrackDLO standalone — frame {frame_idx}',
        width=900, height=600,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='TrackDLO standalone test (no ROS required)'
    )
    p.add_argument('--nodes',        type=int,   default=40,
                   help='Number of rope nodes (default: 40)')
    p.add_argument('--frames',       type=int,   default=20,
                   help='Number of timesteps (default: 20)')
    p.add_argument('--noise',        type=float, default=0.003,
                   help='Point cloud noise std-dev in metres (default: 0.003)')
    p.add_argument('--pts-per-node', type=int,   default=8,
                   help='Dense point samples per segment (default: 8)')
    p.add_argument('--visualize-3d', action='store_true',
                   help='Open3D 3-D visualisation per frame (requires open3d)')
    p.add_argument('--seed',         type=int,   default=42,
                   help='Random seed (default: 42)')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print('=' * 60)
    print('TrackDLO Standalone Test')
    print('=' * 60)
    print(f'  nodes         : {args.nodes}')
    print(f'  frames        : {args.frames}')
    print(f'  noise std     : {args.noise*1000:.1f} mm')
    print(f'  pts per node  : {args.pts_per_node}')
    print()

    # 1. Generate synthetic data
    print('Generating synthetic rope sequence...')
    frames, gt_nodes = make_rope_sequence(
        n_nodes=args.nodes,
        n_frames=args.frames,
        noise_std=args.noise,
        pts_per_node=args.pts_per_node,
    )
    print(f'  {len(frames)} frames, {frames[0].shape[0]} points/frame\n')

    # 2. Run tracker
    print('Running TrackDLO...')
    results, errors, times_ms = run_tracker(frames, gt_nodes, n_nodes=args.nodes)

    # 3. Print summary
    print()
    print('─' * 40)
    print(f'Mean tracking error : {np.mean(errors)*1000:.2f} mm')
    print(f'Max  tracking error : {np.max(errors)*1000:.2f} mm')
    print(f'Mean frame time     : {np.mean(times_ms):.1f} ms')
    print('─' * 40)

    # 4. Optional 3-D frame-by-frame visualisation
    if args.visualize_3d:
        if not HAS_O3D:
            print('[--visualize-3d] open3d not found. Install: pip install open3d')
        else:
            for fi in range(len(results)):
                visualize_3d_frame(frames[fi], gt_nodes[fi], results[fi], fi)

    # 5. Summary plots
    plot_summary(results, gt_nodes, errors, times_ms, args.noise)


if __name__ == '__main__':
    main()
