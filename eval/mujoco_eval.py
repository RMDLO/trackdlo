#!/usr/bin/env python3
"""
eval/mujoco_eval.py
===================
Hardware-free DER-in-MuJoCo evaluation of TrackDLO (no ROS, no camera).

Uses adapteddlo_muj to generate physics-accurate DLO dynamics under gravity
and endpoint manipulation.  The DER node positions serve as ground truth;
synthetic point clouds sampled from segment surfaces are fed to TrackDLO.

Quick start (activate the dlo conda environment first):

    python eval/mujoco_eval.py
    python eval/mujoco_eval.py --frames 30 --noise 0.004
    python eval/mujoco_eval.py --adapteddlo-path /path/to/adapteddlo_muj

Why this script?
    standalone_test.py uses a catenary approximation for the DLO shape.
    This script uses the full Discrete Elastic Rod (DER) model in MuJoCo,
    giving physically correct bending/torsion dynamics and gravity droop.
    Tracking accuracy is therefore measured against a physics-accurate GT,
    not a simplified curve fit.

Dependencies
    - conda activate dlo  (or any env with mujoco >= 3.0 and numpy/matplotlib)
    - adapteddlo_muj built with _Dlo_iso.so (see adapteddlo_muj/notebooks/quickstart.ipynb)
    - trackdlo_cpp.so in pybind_ext/ (already shipped in this repo)
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

# ── locate trackdlo_cpp pybind extension ──────────────────────────────────────
_PYBIND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'pybind_ext')
)
if _PYBIND_DIR not in sys.path:
    sys.path.insert(0, _PYBIND_DIR)

try:
    import trackdlo_cpp
except ImportError as e:
    sys.exit(
        f"Cannot import trackdlo_cpp: {e}\n"
        f"Looked in: {_PYBIND_DIR}\n"
        f"Make sure the .so in pybind_ext/ matches your Python version."
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Hardware-free DER-in-MuJoCo evaluation of TrackDLO'
    )
    p.add_argument(
        '--adapteddlo-path', default=None,
        help='Path to adapteddlo_muj repo root. '
             'Defaults to auto-detecting a sibling directory.'
    )
    p.add_argument('--frames',       type=int,   default=20,
                   help='Evaluation frames (default: 20)')
    p.add_argument('--noise',        type=float, default=0.003,
                   help='Point cloud noise std-dev [m] (default: 0.003)')
    p.add_argument('--pts-per-seg',  type=int,   default=6,
                   help='Point cloud samples per DER segment (default: 6)')
    p.add_argument('--r-len',        type=float, default=0.8,
                   help='Cable arc length [m] (default: 0.8)')
    p.add_argument('--r-pieces',     type=int,   default=30,
                   help='DER segment count (default: 30)')
    p.add_argument('--settle-steps', type=int,   default=800,
                   help='MuJoCo steps to settle before recording (default: 800)')
    p.add_argument('--move-steps',   type=int,   default=300,
                   help='Sim steps between captured frames (default: 300)')
    p.add_argument('--seed',         type=int,   default=42,
                   help='RNG seed (default: 42)')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# DER simulation
# ──────────────────────────────────────────────────────────────────────────────

def _locate_adapteddlo(hint):
    """Return an absolute path to the adapteddlo_muj repo root."""
    if hint is not None:
        path = os.path.abspath(hint)
        if not os.path.isdir(os.path.join(path, 'adapteddlo_muj')):
            sys.exit(f"--adapteddlo-path does not contain adapteddlo_muj/: {path}")
        return path

    # auto-detect: common layouts relative to this file's location
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, '..', '..', 'adapteddlo_muj'),
        os.path.join(this_dir, '..', 'adapteddlo_muj'),
        os.path.join(this_dir, '..', '..', '..', 'adapteddlo_muj'),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'adapteddlo_muj')):
            return os.path.abspath(c)

    sys.exit(
        "Cannot find adapteddlo_muj.\n"
        "Pass --adapteddlo-path /path/to/adapteddlo_muj, or\n"
        "install with: pip install -e /path/to/adapteddlo_muj"
    )


def build_der_sim(r_len, r_pieces, tmpdir, adapteddlo_path):
    """
    Build MuJoCo model + DER controller for a cable in free space.

    Parameters
    ----------
    r_len : float    Cable arc length [m]
    r_pieces : int   DER segment count
    tmpdir : str     Writable directory for generated XML files
    adapteddlo_path : str

    Returns
    -------
    model, data, sim, dlo, site_ids, eef_body_id
    """
    import mujoco
    import adapteddlo_muj as _pkg
    from adapteddlo_muj.assets.genrope.gdv_O_weld2 import GenKin_O_weld2
    from adapteddlo_muj.utils.xml_utils import XMLWrapper
    from adapteddlo_muj.utils.mjc_utils import MjSimWrapper
    import adapteddlo_muj.utils.mjc2_utils as mjc2
    from adapteddlo_muj.controllers.ropekin_controller_adapt import DLORopeAdapt

    r_thickness = 0.010          # 1 cm outer diameter — typical manipulation cable
    alpha_bar   = 5.0e-3         # bending stiffness [N·m²]
    beta_bar    = 3.0e-3         # torsional stiffness [N·m²]

    # ── generate XML ──────────────────────────────────────────────────────────
    rope_path  = os.path.join(tmpdir, 'dlorope1dkin.xml')
    asset_dir  = os.path.join(os.path.dirname(_pkg.__file__), 'assets')
    world_path = os.path.join(asset_dir, 'world_test.xml')
    # GenKin_O_weld2 also writes anchorbox.xml into the same directory as rope_path
    box_path   = os.path.join(tmpdir, 'anchorbox.xml')

    GenKin_O_weld2(
        r_len=r_len,
        r_thickness=r_thickness,
        r_pieces=r_pieces,
        j_stiff=0.0,
        j_damp=1.0,
        init_pos=[r_len / 2, 0.0, 0.5],
        init_quat=[1., 0., 0., 0.],
        d_small=0.,
        rope_type='capsule',
        vis_subcyl=False,
        obj_path=rope_path,
    )

    xml_world = XMLWrapper(world_path)
    xml_world.merge_multiple(XMLWrapper(box_path),  ['worldbody', 'equality', 'contact'])
    xml_world.merge_multiple(XMLWrapper(rope_path), ['worldbody'])
    xml_str = xml_world.get_xml_string()

    # ── build model ───────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_string(xml_str)   # fully in-memory after this
    data  = mujoco.MjData(model)
    sim   = MjSimWrapper(model, data)

    model.opt.gravity[:] = [0.0, 0.0, -9.81]

    # Forward pass BEFORE creating DLORopeAdapt so that site positions are
    # populated when DLORopeAdapt computes rest-state edge lengths (e_bar).
    sim.forward()

    # DER controller — both ends clamped
    dlo = DLORopeAdapt(
        model=model, data=data,
        n_link=r_pieces,
        alpha_bar=alpha_bar,
        beta_bar=beta_bar,
        overall_rot=0.,
        bothweld=True,
    )

    # ── site IDs for DER nodes (S_0 … S_{r_pieces-1}, S_last) ────────────────
    site_ids = [mjc2.obj_name2id(model, 'site', f'S_{i}') for i in range(r_pieces)]
    site_ids.append(mjc2.obj_name2id(model, 'site', 'S_last'))

    # ── endpoint body ID (eef_body = B_last anchor) ────────────────────────────
    eef_body_id = mjc2.obj_name2id(model, 'body', 'eef_body')

    return model, data, sim, dlo, site_ids, eef_body_id


def _step(model, data, sim, dlo):
    """One DER + MuJoCo step: compute torques → integrate → update FK."""
    dlo.update_torque()
    sim.step()
    sim.forward()


def _get_nodes(data, site_ids):
    """Return (N+1, 3) array of DER node world positions (S_0 … S_last order)."""
    return np.array([data.site_xpos[s].copy() for s in site_ids])


# ──────────────────────────────────────────────────────────────────────────────
# Point cloud generation
# ──────────────────────────────────────────────────────────────────────────────

def sample_pointcloud(nodes, pts_per_seg, noise_std, rng):
    """
    Generate a dense point cloud from DER node positions.

    Samples `pts_per_seg` points uniformly along each segment and adds
    isotropic Gaussian noise to simulate camera depth noise.

    Parameters
    ----------
    nodes : (N+1, 3) float64   DER node world positions
    pts_per_seg : int           Points sampled per segment
    noise_std : float           Gaussian noise std-dev [m]
    rng : np.random.Generator

    Returns
    -------
    pts : (N * pts_per_seg, 3) float64
    """
    pts = []
    for i in range(len(nodes) - 1):
        alphas = np.linspace(0, 1, pts_per_seg, endpoint=False)
        for a in alphas:
            pt = (1.0 - a) * nodes[i] + a * nodes[i + 1]
            pt = pt + rng.standard_normal(3) * noise_std
            pts.append(pt)
    return np.array(pts, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# TrackDLO tracking
# ──────────────────────────────────────────────────────────────────────────────

def run_tracking(frames_pts, frames_gt, n_nodes):
    """
    Run trackdlo_cpp on a list of point-cloud frames.

    Parameters
    ----------
    frames_pts : list of (M, 3) arrays   Point clouds (one per frame)
    frames_gt  : list of (n_nodes, 3)    DER ground-truth node positions
                 (site order S_0 … S_last; may be forward or reversed vs tracker)
    n_nodes    : int                     Number of tracker nodes

    Returns
    -------
    tracked  : list of (n_nodes, 3)   Tracked node positions per frame
    errors   : list of float          Mean per-node L2 error vs GT [m]
    times_ms : list of float          Per-frame wall-clock time [ms]
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

    # ── initialise on first frame ──────────────────────────────────────────────
    pts0 = frames_pts[0]
    Y0, sigma2_0 = trackdlo_cpp.reg(pts0, n_nodes, mu=0.05, max_iter=500)
    Y0 = trackdlo_cpp.sort_pts(Y0)
    tracker.initialize_nodes(Y0)

    seg_dis = np.sqrt(np.sum(np.diff(Y0, axis=0) ** 2, axis=1))
    tracker.initialize_geodesic_coord(
        np.concatenate([[0.0], np.cumsum(seg_dis)]).tolist()
    )
    tracker.set_sigma2(sigma2_0)

    # ── align GT ordering with sort_pts output ────────────────────────────────
    # sort_pts picks one end as node 0; the DER site ordering (S_0 … S_last)
    # may run in the opposite direction.  Check once on frame 0 and fix.
    gt0 = frames_gt[0]
    err_fwd = float(np.mean(np.linalg.norm(Y0 - gt0,        axis=1)))
    err_rev = float(np.mean(np.linalg.norm(Y0 - gt0[::-1],  axis=1)))
    if err_rev < err_fwd:
        frames_gt = [gt[::-1] for gt in frames_gt]
        print(f'  [init] GT reversed to match tracker order '
              f'(err_fwd={err_fwd*1e3:.1f} mm vs err_rev={err_rev*1e3:.1f} mm)')

    tracked, errors, times_ms = [], [], []

    for idx, (pts, gt) in enumerate(zip(frames_pts, frames_gt)):
        t0 = time.perf_counter()

        Y      = tracker.get_tracking_result()
        sigma2 = tracker.get_sigma2()

        Y, sigma2, converged = tracker.cpd_lle(
            X=pts, Y=Y, sigma2=sigma2,
            beta=5.0, lambda_=1.0, lle_weight=1.0,
            mu=0.05, max_iter=30, tol=1e-4,
            include_lle=True,
        )

        tracker.initialize_nodes(Y)
        tracker.set_sigma2(sigma2)

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        err        = float(np.mean(np.linalg.norm(Y - gt, axis=1)))

        tracked.append(Y.copy())
        errors.append(err)
        times_ms.append(elapsed_ms)

        print(
            f'  frame {idx:3d}  |  '
            f'error = {err * 1e3:6.2f} mm  |  '
            f'time = {elapsed_ms:6.1f} ms  |  '
            f'{"converged" if converged else "max_iter"}'
        )

    return tracked, errors, times_ms


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(frames_pts, frames_gt, tracked, errors, times_ms, noise_std, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── Panel 1: shape overlay at first / mid / last frame ───────────────────
    ax = axes[0]
    palette = {'first': 'steelblue', 'mid': 'mediumseagreen', 'last': 'tomato'}
    sel     = {'first': 0, 'mid': len(tracked) // 2, 'last': len(tracked) - 1}
    for label, fi in sel.items():
        c  = palette[label]
        gt = frames_gt[fi]
        tr = tracked[fi]
        ax.plot(gt[:, 0], gt[:, 2], '--', color=c, lw=1.5, alpha=0.55,
                label=f'{label} GT')
        ax.plot(tr[:, 0], tr[:, 2], '-o', color=c, lw=2.0, ms=3,
                label=f'{label} tracked')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title(
        'Tracked nodes vs DER ground truth\n'
        '(first / mid / last frame)',
        fontsize=10
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: per-frame tracking error ─────────────────────────────────────
    ax2 = axes[1]
    errs_mm = np.array(errors) * 1e3
    ax2.plot(errs_mm, 'b-o', ms=4, lw=2)
    ax2.axhline(
        errs_mm.mean(), color='r', ls='--',
        label=f'mean = {errs_mm.mean():.2f} mm'
    )
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Mean node error [mm]')
    ax2.set_title(
        f'TrackDLO tracking error per frame\n'
        f'(DER GT, noise σ = {noise_std * 1e3:.1f} mm)',
        fontsize=10
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: computation time ──────────────────────────────────────────────
    ax3 = axes[2]
    ax3.bar(range(len(times_ms)), times_ms, color='steelblue', alpha=0.7)
    ax3.axhline(
        np.mean(times_ms), color='r', ls='--',
        label=f'mean = {np.mean(times_ms):.1f} ms'
    )
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Time [ms]')
    ax3.set_title('TrackDLO computation time per frame', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    print(f'\nFigure saved to {out}')
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # ── resolve adapteddlo_muj ─────────────────────────────────────────────────
    adapteddlo_path = _locate_adapteddlo(args.adapteddlo_path)
    sys.path.insert(0, adapteddlo_path)

    # make _Dlo_iso.so importable (DER C++ extension)
    _dlo_cpp = os.path.join(adapteddlo_path, 'adapteddlo_muj', 'controllers', 'dlo_cpp')
    if _dlo_cpp not in sys.path:
        sys.path.insert(0, _dlo_cpp)

    print('=' * 60)
    print('TrackDLO × adapteddlo_muj — MuJoCo DER Evaluation')
    print('=' * 60)
    print(f'  adapteddlo_muj : {adapteddlo_path}')
    print(f'  cable length   : {args.r_len:.2f} m  ({args.r_pieces} DER segments)')
    print(f'  frames         : {args.frames}')
    print(f'  noise σ        : {args.noise * 1e3:.1f} mm')
    print(f'  pts / segment  : {args.pts_per_seg}')

    # ── build simulation ───────────────────────────────────────────────────────
    print('\n[1/4] Building DER simulation...')
    with tempfile.TemporaryDirectory() as tmpdir:
        model, data, sim, dlo, site_ids, eef_body_id = build_der_sim(
            r_len=args.r_len,
            r_pieces=args.r_pieces,
            tmpdir=tmpdir,
            adapteddlo_path=adapteddlo_path,
        )
    # tmpdir cleaned up — model/data are fully in-memory and unaffected

    n_nodes = args.r_pieces + 1   # S_0 … S_{r_pieces-1} + S_last
    print(f'  Model: {model.nq} qpos, {model.nv} qvel,  '
          f'{model.nbody} bodies, {n_nodes} DER nodes')

    # ── settle under gravity ───────────────────────────────────────────────────
    print(f'\n[2/4] Settling {args.settle_steps} steps under gravity...')
    for _ in range(args.settle_steps):
        _step(model, data, sim, dlo)
    print(f'  Done. Rope centroid z = {_get_nodes(data, site_ids)[:, 2].mean():.3f} m')

    # ── collect frames: sinusoidal z-lift of one endpoint ─────────────────────
    print(f'\n[3/4] Recording {args.frames} frames '
          f'({args.move_steps} sim-steps each)...')

    eef_init_z = float(model.body_pos[eef_body_id, 2])

    frames_pts, frames_gt = [], []
    eef_current_z = eef_init_z    # track current endpoint height for smooth motion

    for fi in range(args.frames):
        # Target: sinusoidal z-lift 0 → 0.20 m → 0 over the full sequence.
        # The endpoint is moved GRADUALLY (delta per step) to avoid exciting
        # high-frequency rope oscillations, making GT well-defined at capture.
        phase = fi / max(args.frames - 1, 1)
        eef_target_z = eef_init_z + 0.20 * np.sin(np.pi * phase)
        dz_per_step  = (eef_target_z - eef_current_z) / max(args.move_steps, 1)

        for _ in range(args.move_steps):
            model.body_pos[eef_body_id, 2] += dz_per_step
            _step(model, data, sim, dlo)

        eef_current_z = float(model.body_pos[eef_body_id, 2])
        gt_nodes = _get_nodes(data, site_ids)
        pts      = sample_pointcloud(gt_nodes, args.pts_per_seg, args.noise, rng)

        frames_gt.append(gt_nodes)
        frames_pts.append(pts)

        if fi == 0 or fi == args.frames - 1 or fi == args.frames // 2:
            print(f'  frame {fi:3d}: {pts.shape[0]} pts, '
                  f'eef z = {eef_current_z:.3f} m')

    # ── run TrackDLO ──────────────────────────────────────────────────────────
    print(f'\n[4/4] Running TrackDLO ({n_nodes} nodes)...')
    tracked, errors, times_ms = run_tracking(frames_pts, frames_gt, n_nodes)

    # ── summary ───────────────────────────────────────────────────────────────
    errs_mm = np.array(errors) * 1e3
    print()
    print('─' * 50)
    print(f'Mean tracking error : {errs_mm.mean():.2f} mm')
    print(f'Max  tracking error : {errs_mm.max():.2f} mm')
    print(f'Mean frame time     : {np.mean(times_ms):.1f} ms')
    print('─' * 50)

    # ── plot ──────────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'mujoco_eval_results.pdf')
    plot_results(frames_pts, frames_gt, tracked, errors, times_ms,
                 args.noise, out)


if __name__ == '__main__':
    main()
