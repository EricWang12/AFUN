import json
import math
import numpy as np
import plotly.graph_objects as go
# Usage (after you compute r, R)
# fig = plotly_curve_frame_viewer(r, R, axis_len=0.25, frame_step=2, show_every=None)
# fig.show()
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

FPS = 30.0
FRAMES = 30


# ----------------------------
# Utils helpers (PyTorch)
# ----------------------------
def fsparams_to_tensor(fsparams: dict, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert 'T0' in fsparams, 'T0 is required'
    assert 'u' in fsparams, 'u is required'
    assert 'v' in fsparams, 'v is required'
    assert 'kappa' in fsparams, 'kappa is required'
    assert 'tau' in fsparams, 'tau is required'
    assert 'velocity' in fsparams, 'velocity is required'

    
    return torch.as_tensor(fsparams['T0'] + [fsparams['u'], fsparams['v'], fsparams['kappa'], fsparams['tau'], fsparams['velocity']], dtype=dtype).contiguous()


def tensor_to_fsparams(tensor: torch.Tensor) -> dict:
    assert tensor.shape[0] == 8, 'tensor must be a 8D vector [ T0[3], u, v, kappa, tau, velocity ]'
    tensor = tensor.float()
    # T0 = tensor[:3].numpy()
    T0 = tensor[:3]
    T0 = T0 / torch.linalg.norm(T0) if torch.linalg.norm(T0) != 0 else T0
    
    
    return {
        'T0': T0.numpy().tolist(),
        'u': tensor[3].numpy(),
        'v': tensor[4].numpy(),
        'kappa': tensor[5].numpy(),
        'tau': tensor[6].numpy(),
        'velocity': tensor[7].numpy(),
    }

# ----------------------------
# SO(3) helpers (PyTorch)
# ----------------------------
def hat(w: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix for cross product: hat(w) x = w × x.  w: (...,3) -> (...,3,3)"""
    wx, wy, wz = w.unbind(dim=-1)
    O = torch.zeros((*w.shape[:-1], 3, 3), dtype=w.dtype, device=w.device)
    O[..., 0, 1] = -wz
    O[..., 0, 2] =  wy
    O[..., 1, 0] =  wz
    O[..., 1, 2] = -wx
    O[..., 2, 0] = -wy
    O[..., 2, 1] =  wx
    return O


def so3_exp(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Rodrigues' formula. w is axis-angle vector (...,3).
    Returns R = exp(hat(w)) (...,3,3).
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # (...,1)
    K = hat(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)

    theta2 = theta * theta
    small = theta < eps

    # A = sin(theta)/theta
    A = torch.where(
        small,
        1 - theta2 / 6 + (theta2 * theta2) / 120,
        torch.sin(theta) / theta,
    )

    # B = (1-cos(theta))/theta^2
    B = torch.where(
        small,
        0.5 - theta2 / 24 + (theta2 * theta2) / 720,
        (1 - torch.cos(theta)) / theta2,
    )

    A = A[..., None]  # (...,1,1)
    B = B[..., None]
    return I + A * K + B * (K @ K)


def make_initial_frame(T0: torch.Tensor, u0: float = 1.0, v0: float = 0.0, kappa: float |None = None, tau: float|None = None, velocity: float |None = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Build an orthonormal TNB frame R0=[T,N,B].
    N is obtained by projecting (u0,v0,0) onto the normal plane of T.
    """
    dev, dt = T0.device, T0.dtype

    T = T0 / torch.linalg.norm(T0)

    n_raw = torch.tensor([u0, v0, 0.0], dtype=dt, device=dev)
    n_proj = n_raw - (n_raw @ T) * T

    if torch.linalg.norm(n_proj) < eps:
        # fallback: pick any vector not parallel to T, then project
        a = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)
        if torch.abs(a @ T) > 0.9:
            a = torch.tensor([0.0, 1.0, 0.0], dtype=dt, device=dev)
        n_proj = a - (a @ T) * T

    N = n_proj / torch.linalg.norm(n_proj)
    B = torch.linalg.cross(T, N, dim=0)

    R0 = torch.stack([T, N, B], dim=1)  # 3x3, columns are T,N,B
    

    kappa = torch.linalg.norm(torch.tensor([u0, v0, 0.0], dtype=dt, device=dev)) if kappa is None else kappa
    tau = 0.0 if tau is None else tau
    velocity = torch.linalg.norm(T0) if velocity is None else velocity
    
    return R0, kappa, tau, velocity



def integrate_curve_se3_time(
    r0: torch.Tensor,
    R0: torch.Tensor,
    kappa: float | torch.Tensor,
    tau: float | torch.Tensor,
    velocity: float | torch.Tensor,
    *,
    frames: int = 30,
    fps: float = 30.0,
):
    N = int(frames)
    if N < 1:
        raise ValueError(f"`frames` must be >= 1, got {frames}.")
    if fps <= 0:
        raise ValueError(f"`fps` must be > 0, got {fps}.")

    dev, dtp = r0.device, r0.dtype
    dt_t = torch.tensor(1.0 / float(fps), dtype=dtp, device=dev)

    def at_i(x, i):
        if torch.is_tensor(x) and x.ndim == 1:
            return x[i]
        return x

    def as_t(x):
        if torch.is_tensor(x):
            return x.to(dtype=dtp, device=dev)
        return torch.tensor(x, dtype=dtp, device=dev)

    r_list = [r0]
    R_list = [R0]

    for i in range(N):
        R_i = R_list[-1]
        r_i = r_list[-1]

        k_i = as_t(at_i(kappa, i))
        t_i = as_t(at_i(tau, i))

        vt_i = as_t(at_i(velocity, i))
        ds_i = vt_i * dt_t

        # FS: w_body = [tau, 0, kappa] in [T,N,B]
        zero = torch.zeros_like(k_i)
        w_body = torch.stack([t_i, zero, k_i])

        R_next = R_i @ so3_exp(w_body * ds_i)

        T = R_i[:, 0]
        r_next = r_i + (vt_i * dt_t) * T

        R_list.append(R_next)
        r_list.append(r_next)

    return torch.stack(r_list, 0), torch.stack(R_list, 0)


def _make_frame_from_tangent(
    T0: torch.Tensor,
    normal_hint: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    T = _safe_normalize(T0, eps=eps)
    if normal_hint is None:
        normal_hint = torch.tensor([0.0, 0.0, 1.0], dtype=T.dtype, device=T.device)
        if float(torch.abs(torch.dot(normal_hint, T)).detach().cpu()) > 0.9:
            normal_hint = torch.tensor([0.0, 1.0, 0.0], dtype=T.dtype, device=T.device)
    else:
        normal_hint = normal_hint.to(dtype=T.dtype, device=T.device)

    M1 = normal_hint - torch.dot(normal_hint, T) * T
    if float(torch.linalg.norm(M1).detach().cpu()) < eps:
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=T.dtype, device=T.device)
        if float(torch.abs(torch.dot(fallback, T)).detach().cpu()) > 0.9:
            fallback = torch.tensor([0.0, 1.0, 0.0], dtype=T.dtype, device=T.device)
        M1 = fallback - torch.dot(fallback, T) * T

    M1 = _safe_normalize(M1, eps=eps)
    M2 = _safe_normalize(torch.linalg.cross(T, M1, dim=0), eps=eps)
    M1 = _safe_normalize(torch.linalg.cross(M2, T, dim=0), eps=eps)
    return torch.stack([T, M1, M2], dim=1)


def _trajectory_linearity_ratio(pts: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    chord = pts[-1] - pts[0]
    chord_len = torch.linalg.norm(chord)
    if float(chord_len.detach().cpu()) < eps:
        return torch.zeros((), dtype=pts.dtype, device=pts.device)

    direction = chord / (chord_len + eps)
    rel = pts - pts[0]
    proj = torch.sum(rel * direction[None, :], dim=1, keepdim=True) * direction[None, :]
    off = rel - proj
    return torch.max(torch.linalg.norm(off, dim=1)) / (chord_len + eps)


def circle_fit_to_fs_params_line_aware(
    trajectory: torch.Tensor | np.ndarray | list | tuple | dict,
    *,
    circle_fit: dict | None = None,
    frames: int | None = None,
    fps: float | None = None,
    linearity_eps: float = 0.08,
    device: str | None = None,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-8,
) -> dict:
    if frames is None:
        frames = FRAMES
    if fps is None:
        fps = FPS
    if device is None:
        device =   "cpu"

    if isinstance(trajectory, dict):
        if "trajectory" not in trajectory:
            raise ValueError("When trajectory is a dict, expected key 'trajectory'.")
        trajectory = [p["position_3d"] for p in trajectory["trajectory"]]

    pts = torch.as_tensor(trajectory, dtype=dtype, device=device)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts must have shape (M,3), got {tuple(pts.shape)}")
    if int(pts.shape[0]) < 2:
        raise ValueError("Need at least 2 points for line-aware FS fitting.")

    M = int(pts.shape[0])
    sample_frames = M - 1
    time_frames = int(frames)
    time_horizon = torch.tensor(float(time_frames) / float(fps), dtype=dtype, device=device)
    p0 = pts[0]
    p1 = pts[-1]

    if circle_fit is None:
        circle_fit = fit_circle_3d_through_endpoints(
            pts,
            device=device,
            dtype=dtype,
        )

    linearity_ratio = _trajectory_linearity_ratio(pts, eps=eps)
    use_line_init = bool((linearity_ratio < linearity_eps).detach().cpu().item())

    chord = p1 - p0
    chord_len = torch.linalg.norm(chord)
    T_line = _safe_normalize(chord, eps=eps)
    polyline_len = torch.sum(torch.linalg.norm(pts[1:] - pts[:-1], dim=1))

    if use_line_init:
        normal_hint = None
        if "normal" in circle_fit:
            normal_hint = circle_fit["normal"]
        R0 = _make_frame_from_tangent(T_line, normal_hint=normal_hint, eps=eps)
        T0 = R0[:, 0]
        N0 = R0[:, 1]
        B0 = R0[:, 2]
        velocity = chord_len / (time_horizon + eps)
        zero = torch.zeros((), dtype=dtype, device=device)
        ret = {
            "T0": T0.detach(),
            "N0": N0.detach(),
            "B0": B0.detach(),
            "R0": R0.detach(),
            "N0_frenet": N0.detach(),
            "kappa": zero.detach(),
            "tau": zero.detach(),
            "velocity": velocity.detach(),
            "linearity_ratio": linearity_ratio.to(dtype=dtype, device=device).detach(),
            "fit_mode": "line_aware" ,
        }
    else:
        ret = circle_to_fs_params(
            circle_fit["center"],
            circle_fit["radius"],
            circle_fit["normal"],
            p0,
            p1=p1,
            p_next=circle_fit["arc"][1],
            frames=time_frames,
            fps=float(fps),
            choose_default_short=circle_fit["choose_default_short"],
        )
        ret["linearity_ratio"] = linearity_ratio.to(dtype=dtype, device=device).detach()
        ret["fit_mode"] =     "circle_fit"
    return ret


# def integrate_curve_se3_time(
#     u,
#     v,
#     omega,
#     r0: torch.Tensor,
#     R0: torch.Tensor,
#     *,
#     frames: int = 30,
#     fps: float = 30.0,
#     velocity: float | torch.Tensor = 1.0,
# ):
#     """
#     Time-parameterized integration with fixed time step dt=1/fps.

#     IMPORTANT: (u,v,omega) are interpreted as *per-arc-length* body rates (i.e. per meter),
#     so the rotation update uses ds = velocity * dt. This ensures changing `velocity` changes
#     only how far the curve advances in space over a fixed number of frames, not the curve's
#     curvature/torsion per unit distance.

#     Updates:
#       ds_i = velocity[i] * dt
#       R_{i+1} = R_i * Exp( hat(w_body) * ds_i )
#       r_{i+1} = r_i + velocity[i] * dt * T_i
#     """
#     N = int(frames)
#     if N < 1:
#         raise ValueError(f"`frames` must be >= 1, got {frames}.")
#     if fps <= 0:
#         raise ValueError(f"`fps` must be > 0, got {fps}.")

#     dev, dtp = r0.device, r0.dtype
#     dt_t = torch.tensor(1.0 / float(fps), dtype=dtp, device=dev)

#     def at_i(x, i):
#         if torch.is_tensor(x) and x.ndim == 1:
#             return x[i]
#         return x

#     def as_t(x):
#         if torch.is_tensor(x):
#             return x.to(dtype=dtp, device=dev)
#         return torch.tensor(x, dtype=dtp, device=dev)

#     r_list = [r0]
#     R_list = [R0]

#     for i in range(N):
#         R_i = R_list[-1]
#         r_i = r_list[-1]

#         om_i = as_t(at_i(omega, i))
#         u_i = as_t(at_i(u, i))
#         v_i = as_t(at_i(v, i))

#         vt_i = as_t(at_i(velocity, i))
#         ds_i = vt_i * dt_t

#         # (u,v,0) are BODY components in the current TNB frame: u along N, v along B
#         w_body = torch.stack([om_i, -v_i, u_i])
#         R_next = R_i @ so3_exp(w_body * ds_i)

#         T = R_i[:, 0]
#         r_next = r_i + (vt_i * dt_t) * T

#         R_list.append(R_next)
#         r_list.append(r_next)

#     r = torch.stack(r_list, dim=0)
#     R = torch.stack(R_list, dim=0)
#     return r, R


def _safe_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    
    return x / (torch.linalg.norm(x) + eps)



def invert_initial_frame_tensor(R: torch.Tensor, eps: float = 1e-8):
    """
    Invert make_initial_frame_tensor in the non-fallback regime.

    Given R=[T,N,B] (columns), return:
      T0 (normalized), u0, v0
    such that projecting (u0,v0,0) onto T⊥ recovers N (up to normalization).
    """
    T = _safe_normalize(R[:, 0])
    N = R[:, 1]

    Tz = T[2]
    if torch.abs(Tz) > eps:
        # Choose n_raw = N + alpha*T with z=0  => alpha = -N_z/T_z
        alpha = -N[2] / Tz
        n_raw = N + alpha * T
        u0, v0 = n_raw[0], n_raw[1]
    else:
        # When Tz≈0, the constructor can only produce N with Nz≈0 (otherwise it would fallback / be impossible).
        u0, v0 = N[0], N[1]

    return T, u0, v0


def make_initial_frame_tensor(
    T0: torch.Tensor,
    u0: torch.Tensor,
    v0: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    R0 = [T,N,B], with N from projecting (u0,v0,0) onto T-perp.
    Keeps u0/v0 in the graph when used during fitting.
    """
    dev, dt = T0.device, T0.dtype
    T = _safe_normalize(T0)

    z = torch.zeros((), dtype=dt, device=dev)
    n_raw = torch.stack([u0, v0, z])
    n_proj = n_raw - (n_raw @ T) * T

    if torch.linalg.norm(n_proj) < eps:
        a = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)
        if torch.abs(a @ T) > 0.99:
            a = torch.tensor([0.0, 1.0, 0.0], dtype=dt, device=dev)
        n_proj = a - (a @ T) * T

    N = _safe_normalize(n_proj)
    B = torch.linalg.cross(T, N, dim=0)
    return torch.stack([T, N, B], dim=1)


def _orthonormal_basis_perp_to(
    d: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit basis (u1,u2) spanning the plane orthogonal to d."""
    d_hat = _safe_normalize(d, eps=eps)
    a = torch.tensor([1.0, 0.0, 0.0], dtype=d.dtype, device=d.device)
    if torch.abs(a @ d_hat) > 0.9:
        a = torch.tensor([0.0, 1.0, 0.0], dtype=d.dtype, device=d.device)
    u1 = _safe_normalize(a - (a @ d_hat) * d_hat, eps=eps)
    u2 = _safe_normalize(torch.linalg.cross(d_hat, u1, dim=0), eps=eps)
    return u1, u2


def _sample_circle_arc(
    c: torch.Tensor,
    r: torch.Tensor,
    n: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    num: int,
    *,
    eps: float = 1e-12,
    choose_default_short: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample the arc from p0 to p1 and return:
      (arc_points [num,3], frames [num,3,3], theta_end, theta_samples)
    """
    e1 = _safe_normalize(p0 - c, eps=eps)
    e2 = _safe_normalize(torch.linalg.cross(n, e1, dim=0), eps=eps)

    v1 = p1 - c
    theta_end = torch.atan2(v1 @ e2, v1 @ e1)

    # Also test the equivalent long-arc endpoint and keep the one that best
    # matches the point ordering from start->end.
    two_pi = torch.tensor(2.0 * np.pi, dtype=r.dtype, device=r.device)
    alt = theta_end - two_pi if float(theta_end.detach().cpu()) >= 0 else theta_end + two_pi
    theta_a = torch.linspace(
        torch.zeros((), dtype=r.dtype, device=r.device),
        theta_end,
        int(num),
        dtype=r.dtype,
        device=r.device,
    )
    theta_b = torch.linspace(
        torch.zeros((), dtype=r.dtype, device=r.device),
        alt,
        int(num),
        dtype=r.dtype,
        device=r.device,
    )

    def arc_from_theta(theta):
        ct = torch.cos(theta)[:, None]
        st = torch.sin(theta)[:, None]
        return c[None, :] + r * (ct * e1[None, :] + st * e2[None, :])

    arc_a = arc_from_theta(theta_a)
    arc_b = arc_from_theta(theta_b)

    # Choose the shorter arc by default; this is usually the desired trajectory
    # branch for start/end-constrained motion segments.
    # choose_a = torch.abs(theta_end) <= torch.abs(alt)
    # choose_a_flag = bool(choose_a.detach().cpu().item())
    theta = theta_a if choose_default_short else theta_b
    theta_target = theta_end if choose_default_short else alt
    arc = arc_a if choose_default_short else arc_b

    # Build a TNB-like frame along the arc.
    # Tangent is d/dtheta normalized and oriented with sequence direction.
    dir_sign = torch.sign(theta_target)
    if float(torch.abs(dir_sign).detach().cpu()) < 1e-12:
        dir_sign = torch.tensor(1.0, dtype=r.dtype, device=r.device)
    T = dir_sign * (-torch.sin(theta)[:, None] * e1[None, :] + torch.cos(theta)[:, None] * e2[None, :])
    N = -(torch.cos(theta)[:, None] * e1[None, :] + torch.sin(theta)[:, None] * e2[None, :])
    B = torch.linalg.cross(T, N, dim=1)
    T = T / (torch.linalg.norm(T, dim=1, keepdim=True) + eps)
    N = N / (torch.linalg.norm(N, dim=1, keepdim=True) + eps)
    B = B / (torch.linalg.norm(B, dim=1, keepdim=True) + eps)
    R = torch.stack([T, N, B], dim=2)
    return arc, R, theta_target, theta


def circle_to_fs_params(
    c: torch.Tensor,
    r: torch.Tensor,
    n: torch.Tensor,
    p0: torch.Tensor,
    *,
    p1: torch.Tensor | None = None,
    p_next: torch.Tensor | None = None,
    choose_default_short: bool = True,
    frames: int = 30,
    fps: float = 30.0,
    eps: float = 1e-12,
) -> dict:
    """
    Convert a 3D circle representation to FS/TNB parameters at the start point.

    The returned frame uses the natural Frenet basis (N0 = N_frenet, pointing
    toward center), so u = kappa and v = 0.  Callers should build R0 from the
    returned T0/N0/B0 (or use the returned ``R0`` key) instead of calling
    ``make_initial_frame(T0, u, v)`` which would produce a different frame.

    Returns dict with keys:
      T0, N0, B0, R0, N0_frenet, kappa, tau, velocity
    """
    n_hat = _safe_normalize(n, eps=eps)
    N0_frenet = _safe_normalize(c - p0, eps=eps)   # principal normal toward center
    kappa = 1.0 / (r + eps)
    tau = torch.zeros((), dtype=r.dtype, device=r.device)

    if p_next is None:
        T0 = _safe_normalize(torch.linalg.cross(n_hat, N0_frenet, dim=0), eps=eps)
    else:
        dp = p_next - p0
        dp_plane = dp - (n_hat @ dp) * n_hat
        dp_tan = dp_plane - (N0_frenet @ dp_plane) * N0_frenet
        if float(torch.linalg.norm(dp_tan).detach().cpu()) < eps:
            T0 = _safe_normalize(torch.linalg.cross(n_hat, N0_frenet, dim=0), eps=eps)
        else:
            T0 = _safe_normalize(dp_tan, eps=eps)

    # Use the natural Frenet frame: N0 = N_frenet (toward center).
    # For a circle (tau=0) this gives u = kappa, v = 0, omega = 0.
    N0 = N0_frenet
    B0 = _safe_normalize(torch.linalg.cross(T0, N0, dim=0), eps=eps)

    # Optional sign convention: align binormal with fitted plane normal.
    # if prefer_n_sign and float((B0 @ n_hat).detach().cpu()) < 0.0:
    #     T0 = -T0
    #     B0 = -B0
    
    

    R0 = torch.stack([T0, N0, B0], dim=1)  # 3x3, columns are T, N, B

    # Choose a constant velocity that reaches the endpoint in the fixed time horizon.
    # With time integration using ds = velocity * dt and (u,v,omega) per-meter, the
    # endpoint is hit exactly iff the endpoint lies on the circle branch; then the
    # required total arc length is S* = r * |theta_target| and velocity = S* / T.
    if p1 is not None:
        if frames < 1:
            raise ValueError(f"frames must be >= 1, got {frames}.")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}.")

        # Compute the signed angle from p0 to p1 in the circle plane basis (e1,e2).
        e1 = _safe_normalize(p0 - c, eps=eps)
        e2 = _safe_normalize(torch.linalg.cross(n_hat, e1, dim=0), eps=eps)
        v1 = p1 - c
        theta_end = torch.atan2(v1 @ e2, v1 @ e1)

        # Consider the equivalent long arc and pick the shorter branch by default.
        two_pi = torch.tensor(2.0 * np.pi, dtype=r.dtype, device=r.device)
        alt = theta_end - two_pi if float(theta_end.detach().cpu()) >= 0 else theta_end + two_pi
        theta_target = theta_end if choose_default_short else alt

        arc_len = torch.abs(theta_target) * r
        total_time = torch.tensor(float(frames) / float(fps), dtype=r.dtype, device=r.device)
        velocity = arc_len / (total_time + eps)
    else:
        velocity = torch.ones((), dtype=r.dtype, device=r.device)

    return {
        "T0": T0,
        "N0": N0,
        "B0": B0,
        "R0": R0,
        "N0_frenet": N0_frenet,
        "kappa": kappa,
        "tau": tau,
        "velocity": velocity,
    }


def circle_to_spline_params(
    c: torch.Tensor,
    r: torch.Tensor,
    n: torch.Tensor,
    p0: torch.Tensor,
    *,
    p1: torch.Tensor,
    choose_default_short: bool = True,
    eps: float = 1e-12,
) -> dict:
    """
    Return a dict containing cubic Bézier control points approximating the circle arc:
      ctrl: (4,3) [P0,P1,P2,P3] with P0=p0, P3=p1
    Also returns R0 consistent with your circle_to_fs_params convention (TNB at start).
    """
    frames, fps = FRAMES, FPS
    dev, dt = p0.device, p0.dtype

    # --- FS-like initial frame (matches your circle_to_fs_params) ---
    n_hat = _safe_normalize(n, eps)


    # --- signed angle on circle plane basis (e1,e2) ---
    e1 = _safe_normalize(p0 - c, eps)                            # center->start
    e2 = _safe_normalize(torch.linalg.cross(n_hat, e1, dim=0), eps)
    v1 = p1 - c
    theta_end = torch.atan2(torch.dot(v1, e2), torch.dot(v1, e1))  # (-pi, pi]

    two_pi = torch.as_tensor(2.0 * math.pi, dtype=dt, device=dev)
    alt = theta_end - two_pi if float(theta_end.detach().cpu()) >= 0.0 else theta_end + two_pi
    theta = theta_end if choose_default_short else alt            # signed target angle

    # --- cubic Bézier circular-arc approximation ---
    # alpha = 4/3 * tan(|theta|/4)
    alpha = (4.0 / 3.0) * torch.tan(torch.abs(theta) / 4.0)

    # Unit tangents consistent with traversal direction (sign(theta))
    s = torch.where(theta >= 0, torch.ones((), dtype=dt, device=dev), -torch.ones((), dtype=dt, device=dev))
    T_start = s * e2
    T_end = s * _safe_normalize((-torch.sin(theta)) * e1 + (torch.cos(theta)) * e2, eps)

    P0 = p0
    P3 = p1
    P1 = P0 + (r * alpha) * T_start
    P2 = P3 - (r * alpha) * T_end
    ctrl = torch.stack([P0, P1, P2, P3], dim=0)                  # (4,3)
    

    # Make control points relative to p0
    ctrl = ctrl - p0  # (4,3) control points relative to p0

    return {
        "ctrl": ctrl[1:],
        "r0": p0,
        "theta": theta,
    }


def fit_circle_3d_through_endpoints(
    trajectory: torch.Tensor | np.ndarray | list | tuple | dict,
    *,
    iters: int = 1500,
    lr: float = 5e-2,
    robust: bool = True,
    huber_delta_scale: float = 0.02,
    t_scale: float = 1.0,
    device: str | None = None,
    dtype: torch.dtype = torch.float64,
    verbose_every: int | None = None,
    streak_early_stop: int = 20,
    early_stop_threshold: float = 2e-8,
):
    """
    Fit a constrained 3D circle to trajectory points with hard endpoint pass-through.

    Geometry constraints:
      - Circle must pass through first/last points.
      - Circle plane must contain the endpoint chord.
      - Circle center lies on the chord perpendicular-bisector plane.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(trajectory, dict):
        if "trajectory" not in trajectory:
            raise ValueError("When trajectory is a dict, expected key 'trajectory'.")
        trajectory = [p["position_3d"] for p in trajectory["trajectory"]]

    pts = torch.as_tensor(trajectory, dtype=dtype, device=device)
  
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts must have shape (M,3), got {tuple(pts.shape)}")
    if int(pts.shape[0]) < 3:
        raise ValueError("Need at least 3 points to fit a circle.")

    p0 = pts[0]
    p1 = pts[-1]
    d = p1 - p0
    L = torch.linalg.norm(d)
    if float(L.detach().cpu()) < 1e-9:
        raise ValueError("Start/end points are too close; constrained circle is underdetermined.")

    # 1) Best-fit plane normal constrained by n·d=0 and plane through p0.
    X = pts - p0                   # X: coordinates of points relative to p0, shape (M,3)
    A = X.transpose(0, 1) @ X      # A: covariance matrix of points about p0, shape (3,3)
    u1, u2 = _orthonormal_basis_perp_to(d) # u1, u2: two orthonormal vectors perpendicular to d (chord direction), each shape (3,)
    U = torch.stack([u1, u2], dim=1)    # U: matrix with u1, u2 as columns, shape (3,2); an orthonormal basis for the plane perpendicular to d
    B2 = U.transpose(0, 1) @ A @ U      # B2: covariance matrix A projected into the plane perpendicular to d, shape (2,2)
    evals, evecs = torch.linalg.eigh(B2) # evals: eigenvalues, evecs: eigenvectors of B2
    alpha = evecs[:, torch.argmin(evals)] # alpha: eigenvector of B2 associated with the smallest eigenvalue, shape (2,)
    n = _safe_normalize(U @ alpha)       # n: best-fit plane normal vector constrained orthogonal to d, shape (3,)

    # 2) Center line c(t) = m + t*u, with u along intersection of circle plane
    # and perpendicular-bisector plane.
    m = 0.5 * (p0 + p1)
    u_line = torch.linalg.cross(d, n, dim=0)
    nu = torch.linalg.norm(u_line)
    if float(nu.detach().cpu()) < 1e-12:
        raise ValueError("Degenerate fit: d x n is near zero (points nearly collinear).")
    u_line = u_line / nu

    proj = X @ n
    Q = pts - proj[:, None] * n[None, :]

    # 3) 1D optimization over center-line coordinate t.
    t_raw = torch.nn.Parameter(torch.zeros((), dtype=dtype, device=device))
    opt = torch.optim.Adam([t_raw], lr=lr)
    loss_hist = []
    no_change_streak = 0
    delta = torch.as_tensor(huber_delta_scale, dtype=dtype, device=device) * L
    t_scale_t = torch.as_tensor(max(float(t_scale), 1e-6), dtype=dtype, device=device) * L
    choose_default_short = True

    for it in range(int(iters)):
        opt.zero_grad(set_to_none=True)
        t = t_scale_t * torch.sinh(t_raw)
        c = m + t * u_line
        r = torch.linalg.norm(c - p0)

        # --- distance-to-ARC residual: evaluate both short + long arcs, then soft-min ---
        e1 = _safe_normalize(p0 - c)  # reference direction: center -> start
        e2 = _safe_normalize(torch.linalg.cross(n, e1, dim=0))  # in-plane orthonormal

        v1 = (p1 - c)
        theta_end = torch.atan2(torch.dot(v1, e2), torch.dot(v1, e1))  # in (-pi, pi]
        flip = (theta_end < 0)
        e2 = torch.where(flip, -e2, e2)
        theta_end = torch.abs(theta_end)  # in [0, pi]

        v = Q - c[None, :]  # (M,3)
        x = v @ e1          # (M,)
        y = v @ e2          # (M,)
        theta = torch.atan2(y, x)  # (M,) in (-pi, pi]

        two_pi = r.new_tensor(2.0 * math.pi)
        theta_long_end = theta_end - two_pi  # in [-2pi, -pi]

        theta_s = torch.clamp(theta, min=0.0, max=theta_end.detach().cpu().item())
        closest_s = c[None, :] + r * (
            torch.cos(theta_s)[:, None] * e1[None, :] +
            torch.sin(theta_s)[:, None] * e2[None, :]
        )
        e_s = torch.linalg.norm(Q - closest_s, dim=1)

        # long arc: map angles to (-2pi, 0] then clamp to [theta_end-2pi, 0]
        theta_m = torch.where(theta > 0, theta - two_pi, theta)  # (-2pi, 0]
        theta_l = torch.clamp(theta_m, min=theta_long_end, max=torch.tensor(0.0, dtype=dtype, device=device))
        closest_l = c[None, :] + r * (
            torch.cos(theta_l)[:, None] * e1[None, :] +
            torch.sin(theta_l)[:, None] * e2[None, :]
        )
        e_l = torch.linalg.norm(Q - closest_l, dim=1)

        if robust:
            delta_f = float(delta.detach().cpu().item())
            loss_s = F.huber_loss(e_s, torch.zeros_like(e_s), delta=delta_f, reduction="mean")
            loss_l = F.huber_loss(e_l, torch.zeros_like(e_l), delta=delta_f, reduction="mean")
        else:
            loss_s = (e_s * e_s).mean()
            loss_l = (e_l * e_l).mean()

        choose_default_short = bool((loss_s < loss_l).detach().cpu().item())

        tau = torch.clamp(1e-2 * (L.detach() ** 2), min=1e-12)
        loss = -tau * torch.logsumexp(torch.stack([-loss_s / tau, -loss_l / tau]), dim=0)

        loss.backward()
        opt.step()
        loss_hist.append(float(loss.detach().cpu()))
        no_change_streak = no_change_streak + 1 if len(loss_hist) > 1 and abs(loss_hist[-1] - loss_hist[-2]) < early_stop_threshold else 0
        if no_change_streak >= int(streak_early_stop):
            print(f"[{it:04d}] circle_loss={loss_hist[-1]:.6e} t={float(t.detach().cpu()):+.6f}")
            print("Early stop due to no change streak.")
            break

        if verbose_every is not None and (it % int(verbose_every) == 0 or it == int(iters) - 1):
            print(f"[{it:04d}] circle_loss={loss_hist[-1]:.6e} t={float(t.detach().cpu()):+.6f}")

    with torch.no_grad():
        t_star = t_scale_t * torch.sinh(t_raw)
        c = m + t_star * u_line
        r = torch.linalg.norm(c - p0)
        arc, R_fit, theta_end, theta_samples = _sample_circle_arc(
            c, r, n, p0, p1, int(pts.shape[0]), choose_default_short=choose_default_short
        )

    return {
        "center": c,
        "radius": r,
        "normal": n,
        "arc": arc,
        "R_fit": R_fit,
        "choose_default_short": choose_default_short,
    }


def fit_circle_3d_through_endpoints_to_fs_params(
    trajectory: torch.Tensor | np.ndarray | list | tuple | dict,
    *,
    circle_fit: dict | None = None,
    iters: int = 1500,
    lr: float = 5e-2,
    robust: bool = True,
    huber_delta_scale: float = 0.02,
    t_scale: float = 1.0,
    device: str | None = None,
    dtype: torch.dtype = torch.float64,
    verbose_every: int | None = None,
    streak_early_stop: int = 20,
):
    """
    Fit a constrained 3D circle to trajectory points with hard endpoint pass-through,
    then refine with a Bishop/RMF model before converting to FS/TNB parameters.

    Geometry constraints:
      - Circle must pass through first/last points.
      - Circle plane must contain the endpoint chord.
      - Circle center lies on the chord perpendicular-bisector plane.

    Returns dict with circle geometry and refined FS-compatible parameters.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(trajectory, dict):
        if "trajectory" not in trajectory:
            raise ValueError("When trajectory is a dict, expected key 'trajectory'.")
        trajectory = [p["position_3d"] for p in trajectory["trajectory"]]

    pts = torch.as_tensor(trajectory, dtype=dtype, device=device)
    fit = circle_fit
    if fit is None:
        fit = fit_circle_3d_through_endpoints(
            pts,
            iters=iters,
            lr=lr,
            robust=robust,
            huber_delta_scale=huber_delta_scale,
            t_scale=t_scale,
            verbose_every=verbose_every,
            streak_early_stop=streak_early_stop,
        )

    c = fit["center"]
    r = fit["radius"]
    n = fit["normal"]
    arc = fit["arc"]
    refined_fit = circle_fit_to_fs_params_line_aware(
        pts,
        circle_fit=fit,
        frames=int(pts.shape[0]) - 1,
        fps=float(max(int(pts.shape[0]) - 1, 1)),
        device=device,
        dtype=dtype,
    )

    return {
        "center": c.detach().cpu(),
        "radius": r.detach().cpu(),
        "normal": n.detach().cpu(),
        "kappa": refined_fit["kappa"].detach().cpu(),
        "tau": refined_fit["tau"].detach().cpu(),
        "velocity": refined_fit["velocity"].detach().cpu(),
        "initial_T": refined_fit["T0"].detach().cpu(),
        "initial_R0": refined_fit["R0"].detach().cpu(),
        "r_fit": arc.detach().cpu(),
        "R_fit": fit["R_fit"].detach().cpu(),
        "fit_mode": refined_fit["fit_mode"],
        "linearity_ratio": refined_fit["linearity_ratio"],
        
    }

# ----------------------------
# Plotting (static + animation)
# ----------------------------
def plot_curve_with_frames(r: torch.Tensor, R: torch.Tensor, stride: int = 25, axis_len: float = 0.2):
    """
    Static plot: curve + a few frames.
    """
    rr = r.detach().cpu().numpy()
    RR = R.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rr[:, 0], rr[:, 1], rr[:, 2], linewidth=2)

    for i in range(0, len(rr), stride):
        p = rr[i]
        T = RR[i, :, 0]
        U = RR[i, :, 1]
        V = RR[i, :, 2]
        ax.plot([p[0], p[0] + axis_len * T[0]], [p[1], p[1] + axis_len * T[1]], [p[2], p[2] + axis_len * T[2]])
        ax.plot([p[0], p[0] + axis_len * U[0]], [p[1], p[1] + axis_len * U[1]], [p[2], p[2] + axis_len * U[2]])
        ax.plot([p[0], p[0] + axis_len * V[0]], [p[1], p[1] + axis_len * V[1]], [p[2], p[2] + axis_len * V[2]])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Curve + moving frame [T,U,V]")
    plt.tight_layout()
    plt.show()


def animate_curve_with_frame(r: torch.Tensor, R: torch.Tensor, axis_len: float = 0.3, fps: int = 30, out_gif: str = None):
    """
    Animated plot: curve + a single moving frame at the current point.
    If out_gif is not None, saves a GIF using PillowWriter.
    """
    rr = r.detach().cpu().numpy()
    RR = R.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rr[:, 0], rr[:, 1], rr[:, 2], linewidth=2)

    # moving point
    pt, = ax.plot([rr[0, 0]], [rr[0, 1]], [rr[0, 2]], marker="o", linestyle="None")

    # moving axes (3 line segments)
    lT, = ax.plot([], [], [])
    lU, = ax.plot([], [], [])
    lV, = ax.plot([], [], [])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Animated moving frame [T,U,V]")

    def set_line(line, p, d):
        line.set_data([p[0], p[0] + axis_len * d[0]], [p[1], p[1] + axis_len * d[1]])
        line.set_3d_properties([p[2], p[2] + axis_len * d[2]])

    def update(i):
        p = rr[i]
        T = RR[i, :, 0]
        U = RR[i, :, 1]
        V = RR[i, :, 2]

        pt.set_data([p[0]], [p[1]])
        pt.set_3d_properties([p[2]])

        set_line(lT, p, T)
        set_line(lU, p, U)
        set_line(lV, p, V)
        return pt, lT, lU, lV

    ani = FuncAnimation(fig, update, frames=len(rr), interval=1000 / fps, blit=False)

    if out_gif is not None:
        ani.save(out_gif, writer=PillowWriter(fps=fps))

    plt.show()




def plotly_curve_frame_viewer(r, R, axis_len=0.25, frame_step=2, show_every=None, u=0.0, v=0.0, omega=0.0):
    rr = r.detach().cpu().numpy()
    RR = R.detach().cpu().numpy()

    Ntot = rr.shape[0]
    idxs = np.arange(0, Ntot, frame_step, dtype=int)
    if idxs[-1] != Ntot - 1:
        idxs = np.append(idxs, Ntot - 1)

    curve_rr = rr if show_every is None else rr[::show_every]

    def at_i(x, i):
        if hasattr(x, "detach"):  # torch tensor
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray) and x.ndim == 1:
            return float(x[i])
        return float(x)

    def seg(p, d):
        q = p + axis_len * d
        return np.array([p[0], q[0]]), np.array([p[1], q[1]]), np.array([p[2], q[2]])

    def plane_mesh(p, Nvec, Bvec, s):
        # square in the normal plane spanned by N,B
        c0 = p + s*( Nvec + Bvec)
        c1 = p + s*( Nvec - Bvec)
        c2 = p + s*(-Nvec - Bvec)
        c3 = p + s*(-Nvec + Bvec)
        V = np.stack([c0,c1,c2,c3], axis=0)
        i = [0, 0]
        j = [1, 2]
        k = [2, 3]
        return go.Mesh3d(x=V[:,0], y=V[:,1], z=V[:,2], i=i, j=j, k=k, name="normal_plane", opacity=0.25)

    def frame_geom(i):
        p = rr[i]
        T = RR[i, :, 0]
        Nvec = RR[i, :, 1]
        Bvec = RR[i, :, 2]

        ui, vi, omi = at_i(u, i), at_i(v, i), at_i(omega, i)

        # interpret (u,v,0) as BODY coords in current TNB -> world vector u*N + v*B
        uv_vec = ui * Nvec + vi * Bvec

        omega_vec = omi * Bvec

        xT,  yT,  zT  = seg(p, T)
        xuv, yuv, zuv = seg(p, uv_vec)
        xOm, yOm, zOm = seg(p, omega_vec)

        plane = plane_mesh(p, Nvec, Bvec, s=axis_len * 0.7)

        return p, plane, (xT,yT,zT), (xuv,yuv,zuv), (xOm,yOm,zOm)

    # Initial
    i0 = idxs[0]
    p0, plane0, (xT,yT,zT), (xuv,yuv,zuv), (xOm,yOm,zOm) = frame_geom(i0)

    traj = go.Scatter3d(x=curve_rr[:,0], y=curve_rr[:,1], z=curve_rr[:,2], mode="lines", name="curve")
    pt   = go.Scatter3d(x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="markers", marker=dict(size=4), name="point")

    lT   = go.Scatter3d(x=xT,  y=yT,  z=zT,  mode="lines", name="T")
    luv0 = go.Scatter3d(x=xuv, y=yuv, z=zuv, mode="lines", name="(u,v,0) in NB")
    lOm  = go.Scatter3d(x=xOm, y=yOm, z=zOm, mode="lines", name="omega (along B)")

    frames = []
    slider_steps = []
    for i in idxs:
        p, plane, (xT,yT,zT), (xuv,yuv,zuv), (xOm,yOm,zOm) = frame_geom(i)
        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    traj,
                    go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", marker=dict(size=4)),
                    plane,
                    go.Scatter3d(x=xT,  y=yT,  z=zT,  mode="lines"),
                    go.Scatter3d(x=xuv, y=yuv, z=zuv, mode="lines"),
                    go.Scatter3d(x=xOm, y=yOm, z=zOm, mode="lines"),
                ],
                layout=go.Layout(title_text=f"i={i} / {Ntot-1}")
            )
        )
        slider_steps.append(dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=str(i)
        ))

    fig = go.Figure(
        data=[traj, pt, plane0, lT, luv0, lOm],
        frames=frames,
        layout=go.Layout(
            title=f"i={i0} / {Ntot-1}",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.02, y=1.05,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=33, redraw=True), transition=dict(duration=0),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(x=0.15, y=1.02, len=0.8, currentvalue=dict(prefix="index: "), steps=slider_steps)]
        )
    )
    return fig


def _depth_to_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray | None,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    stride: int = 4,
    z_min: float = 0.01,
    z_max: float = 3.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Lift a depth image (meters) into a camera-frame point cloud.

    Returns:
      xyz: (N,3) float array in meters
      colors: (N,3) uint8 array if rgb is provided, else None
    """
    if depth_m.ndim != 2:
        raise ValueError(f"`depth_m` must be (H,W), got shape {depth_m.shape}.")
    if stride < 1:
        raise ValueError(f"`stride` must be >= 1, got {stride}.")

    h, w = depth_m.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[vv, uu]

    valid = np.isfinite(z) & (z > float(z_min)) & (z < float(z_max))
    uu = uu[valid]
    vv = vv[valid]
    z = z[valid]

    x = (uu - float(cx)) * z / float(fx)
    y = (vv - float(cy)) * z / float(fy)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)

    if rgb is None:
        return xyz, None

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"`rgb` must be (H,W,3), got shape {rgb.shape}.")
    if rgb.shape[0] != h or rgb.shape[1] != w:
        raise ValueError(f"`depth_m` shape {depth_m.shape} and `rgb` shape {rgb.shape} do not match.")

    colors = rgb[vv, uu]
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return xyz, colors



def sample_fs_curve(
    fsparams: torch.Tensor | dict,
    T: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Sample ``T`` positions along an FSFrame-integrated curve starting at the origin.

    Parameters
    ----------
    fsparams:
        Either an 8-dim tensor ``[T0(3), u, v, kappa, tau, velocity]`` (from
        :func:`fsparams_to_tensor`) or the equivalent dict.
    T:
        Number of sample points along the curve (>=1). Integration uses
        ``frames=T-1, fps=T-1`` so ``dt=1/(T-1)`` and total time = 1.0.

    Returns
    -------
    torch.Tensor of shape (T, 3) starting at ``r0 = 0``.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1; got {T}")

    if isinstance(fsparams, dict):
        fs_t = fsparams_to_tensor(fsparams, dtype=dtype)
    else:
        fs_t = torch.as_tensor(fsparams, dtype=dtype)
    if fs_t.numel() != 8:
        raise ValueError(f"fsparams must have 8 elements; got {tuple(fs_t.shape)}")

    if device is None:
        device = fs_t.device
    fs_t = fs_t.to(device=device, dtype=dtype)

    T0 = fs_t[:3]
    u = float(fs_t[3])
    v = float(fs_t[4])
    kappa = fs_t[5]
    tau = fs_t[6]
    velocity = fs_t[7]

    if torch.linalg.norm(T0) < 1e-12:
        T0 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    R0, kappa_t, tau_t, velocity_t = make_initial_frame(
        T0, u0=u, v0=v, kappa=kappa, tau=tau, velocity=velocity,
    )

    r0 = torch.zeros(3, dtype=dtype, device=device)
    frames = max(int(T) - 1, 1)
    fps = float(max(int(T) - 1, 1))
    r_curve, _ = integrate_curve_se3_time(
        r0, R0, kappa_t, tau_t, velocity_t,
        frames=frames, fps=fps,
    )
    # integrate_curve_se3_time returns (frames+1, 3) = (T, 3) when T>=2.
    if r_curve.shape[0] != int(T):
        # T == 1 edge case: just return r0.
        r_curve = r_curve[:int(T)]
    return r_curve


def compute_fsframe_curve(
    fsparams: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an FSFrame curve from *fsparams* and return positions + frames.

    Parameters
    ----------
    fsparams : dict
        Expected keys: r0, T0, u, v.
        Optional: kappa, tau, velocity.

    Returns
    -------
    r_curve_np : np.ndarray, shape (frames+1, 3)
        Curve positions.
    R_curve_np : np.ndarray, shape (frames+1, 3, 3)
        Frame (rotation) matrices along the curve.
    """
    r0 = fsparams.get("r0", None)
    T0 = fsparams.get("T0", None)
    kappa = fsparams.get("kappa", None)
    tau = fsparams.get("tau", None)
    velocity = fsparams.get("velocity", None)
    u = float(fsparams.get("u", 1.0))
    v = float(fsparams.get("v", 0.0))
    frames = FRAMES
    fps = FPS

    if r0 is None:
        r0_t = torch.zeros(3, dtype=torch.float64, device="cpu")
    elif torch.is_tensor(r0):
        r0_t = r0.to(dtype=torch.float64, device="cpu")
    else:
        r0_t = torch.tensor(np.asarray(r0), dtype=torch.float64, device="cpu")

    if T0 is None:
        T0_t = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device="cpu")
    elif torch.is_tensor(T0):
        T0_t = T0.to(dtype=torch.float64, device="cpu")
    else:
        T0_t = torch.tensor(np.asarray(T0), dtype=torch.float64, device="cpu")

    if torch.linalg.norm(T0_t) < 1e-12:
        T0_t = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device="cpu")

    R0_t, kappa_t, tau_t, velocity_t = make_initial_frame(
        T0_t, u0=float(u), v0=float(v),
        kappa=kappa, tau=tau, velocity=velocity,
    )

    r_curve, R_curve = integrate_curve_se3_time(
        r0_t, R0_t, kappa_t, tau_t, velocity_t,
        frames=int(frames), fps=float(fps),
    )

    r_curve_np = r_curve.detach().cpu().numpy()
    R_curve_np = R_curve.detach().cpu().numpy()
    return r_curve_np, R_curve_np



# ----------------------------
# 2) spline ctrl pts -> sampled curve (r, R)
# ----------------------------
def _binomial_coefficients(n: int) -> list[int]:
    vals = [1]
    for k in range(1, n + 1):
        vals.append(vals[-1] * (n - k + 1) // k)
    return vals


def compute_spline_curve(
    spline_params_or_ctrl: dict | torch.Tensor | np.ndarray,
    *,
    frames: int | None = None,
    fps: float | None = None,     # kept for parity; not needed for geometry sampling
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given Bézier control points of any degree K>=2, sample positions r and frames R.

    ``spline_params_or_ctrl`` accepts either:
      * a dict {"r0": (3,), "ctrl": (K,3) or (3K,) relative control points}.
        The curve is built with P0 = r0 and the K control points offset by r0.
      * a raw tensor/array of absolute control points shaped (K+1, 3).

    Returns:
      r_curve_np: (frames+1, 3)
      R_curve_np: (frames+1, 3, 3) with columns [T, N, B]
    """

    def _rodrigues_rotate(v: torch.Tensor, k_hat: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        # v, k_hat: (3,), angle: scalar
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        return v * ca + torch.linalg.cross(k_hat, v) * sa + k_hat * (torch.dot(k_hat, v)) * (1.0 - ca)

    # --- parse inputs: resolve to (K+1, 3) absolute control points ---
    if isinstance(spline_params_or_ctrl, dict):
        ctrl = spline_params_or_ctrl["ctrl"]
        ctrl = torch.as_tensor(ctrl)
        if ctrl.ndim == 1:
            if ctrl.numel() % 3 != 0:
                raise ValueError(
                    f"Flat ctrl must have length divisible by 3; got {ctrl.numel()}"
                )
            ctrl = ctrl.reshape(ctrl.numel() // 3, 3)
        if ctrl.ndim != 2 or ctrl.shape[-1] != 3:
            raise ValueError(f"ctrl must reshape to (K,3); got {tuple(ctrl.shape)}")
        r0 = torch.as_tensor(spline_params_or_ctrl["r0"], device=ctrl.device)
        zero = torch.zeros_like(ctrl[0:1])                      # (1, 3)
        ctrl_rel = torch.cat([zero, ctrl], dim=0)               # (K+1, 3)
        ctrl = ctrl_rel + r0[None, :]
    else:
        ctrl = torch.as_tensor(spline_params_or_ctrl)

    if ctrl.ndim != 2 or ctrl.shape[-1] != 3 or ctrl.shape[0] < 3:
        raise ValueError(
            f"compute_spline_curve requires ctrl of shape (K+1, 3) with K+1 >= 3; got {tuple(ctrl.shape)}"
        )
    n = int(ctrl.shape[0]) - 1                                   # Bezier degree

    if frames is None:
        frames = FRAMES
    if fps is None:
        fps = FPS

    ctrl64 = ctrl.to(dtype=torch.float64)

    # Precomputed binomial coeffs
    binom = torch.tensor(_binomial_coefficients(n), dtype=torch.float64, device=ctrl.device)

    ts = torch.linspace(0.0, 1.0, frames + 1, dtype=torch.float64, device=ctrl.device)

    def _bernstein_basis(tt: torch.Tensor, deg: int) -> torch.Tensor:
        # Return (deg+1, T) matrix of Bernstein weights with exact endpoint handling.
        T_pts = tt.shape[0]
        ks = torch.arange(deg + 1, dtype=torch.float64, device=tt.device)
        log_t = torch.log(tt.clamp_min(1e-30))
        log_um = torch.log((1.0 - tt).clamp_min(1e-30))
        binom_d = torch.tensor(_binomial_coefficients(deg), dtype=torch.float64, device=tt.device)
        log_B = binom_d.log()[:, None] + ks[:, None] * log_t[None, :] + (deg - ks)[:, None] * log_um[None, :]
        B = log_B.exp()
        # Enforce exact endpoint values (0^0 convention).
        B[:, 0] = 0.0
        B[0, 0] = 1.0
        B[:, -1] = 0.0
        B[-1, -1] = 1.0
        return B

    B_n = _bernstein_basis(ts, n)                               # (n+1, T)
    r_curve = (B_n.T @ ctrl64)                                   # (T, 3)

    # Derivative: Q_k = n*(P_{k+1} - P_k), a degree-(n-1) Bezier.
    Q = n * (ctrl64[1:] - ctrl64[:-1])                           # (n, 3)
    B_nm1 = _bernstein_basis(ts, n - 1) if n >= 1 else None
    d1 = (B_nm1.T @ Q) if B_nm1 is not None else torch.zeros_like(r_curve)

    # 2nd derivative for the seed frame: Q2_k = (n-1)*(Q_{k+1}-Q_k), degree (n-2).
    if n >= 2:
        Q2 = (n - 1) * (Q[1:] - Q[:-1])                          # (n-1, 3)
        B_nm2 = _bernstein_basis(ts[:1], n - 2)                  # (n-1, 1) — only need d2[0]
        d2_0 = (B_nm2.T @ Q2).squeeze(0)                         # (3,)
    else:
        d2_0 = torch.zeros(3, dtype=torch.float64, device=ctrl.device)
    d2 = torch.zeros_like(d1)
    d2[0] = d2_0

    # Unify with legacy variable names below.
    P0 = ctrl64[0]
    P3 = ctrl64[-1]

    # --- tangents ---
    T = torch.stack([_safe_normalize(d1[i], eps) for i in range(d1.shape[0])], dim=0)  # (N,3)


    b_raw = torch.linalg.cross(d1[0], d2[0])
    if float(torch.linalg.norm(b_raw).detach().cpu()) < eps:
        b_raw = torch.linalg.cross(d1[0], (P3 - P0))
    if float(torch.linalg.norm(b_raw).detach().cpu()) < eps:
        # final fallback: pick any axis not parallel to T0
        a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        if float(torch.abs(torch.dot(a, T[0])).detach().cpu()) > 0.9:
            a = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        b_raw = torch.linalg.cross(T[0], a)
    B0 = _safe_normalize(b_raw, eps)

    # make B0 ⟂ T0
    B0 = _safe_normalize(B0 - torch.dot(B0, T[0]) * T[0], eps)
    N0 = _safe_normalize(torch.linalg.cross(B0, T[0]), eps)
    B0 = _safe_normalize(torch.linalg.cross(T[0], N0), eps)

    # --- parallel transport frames ---
    N = torch.empty_like(T)
    B = torch.empty_like(T)
    N[0], B[0] = N0, B0

    for i in range(1, T.shape[0]):
        t_prev = T[i - 1]
        t_cur = T[i]
        axis = torch.linalg.cross(t_prev, t_cur)
        s = torch.linalg.norm(axis)

        if float(s.detach().cpu()) < eps:
            b_cur = B[i - 1]
        else:
            axis_hat = axis / s
            angle = torch.atan2(s, torch.dot(t_prev, t_cur))
            b_cur = _rodrigues_rotate(B[i - 1], axis_hat, angle)

        # re-orthonormalize
        b_cur = _safe_normalize(b_cur - torch.dot(b_cur, t_cur) * t_cur, eps)
        n_cur = _safe_normalize(torch.linalg.cross(b_cur, t_cur), eps)
        b_cur = _safe_normalize(torch.linalg.cross(t_cur, n_cur), eps)  # enforce right-handed

        B[i] = b_cur
        N[i] = n_cur

    R_curve = torch.stack([T, N, B], dim=-1)       # (N,3,3) columns [T,N,B]

    return r_curve.detach().cpu().numpy(), R_curve.detach().cpu().numpy()


def plotly_pointcloud_and_curve(
    depth: np.ndarray,
    rgb: np.ndarray | object | None,
    *,
    cam_intrinsic: dict,
    r_curve: np.ndarray | None,
    R_curve: np.ndarray | None,
    frame_axis_len: float = 0.08,
    title: str = "Point Cloud + Motion curve",
) -> "go.Figure":
    """
    Plotly visualization: depth point cloud + precomputed FSFrame curve.

    Parameters
    ----------
    depth : np.ndarray, shape (H, W)
        Depth map in metres (auto-scaled if values > 100).
    rgb : np.ndarray (H,W,3) | PIL-image-like | None
        Colour per pixel; ``None`` falls back to a uniform grey.
    cam_intrinsic : dict
        Must contain fx, fy, cx, cy.
        Optional: scene_stride, z_min, z_max, max_points.
    r_curve : np.ndarray, shape (N, 3) or None
        Curve positions (from :func:`compute_fsframe_curve`). If None, only the
        point cloud is rendered (e.g., for OPD vis where the motion is axis+origin).
    R_curve : np.ndarray, shape (N, 3, 3) or None
        Frame matrices (from :func:`compute_fsframe_curve`). If None, the end-tangent
        cone is skipped.
    frame_axis_len : float
        Length of the arrow-head cone at the curve tip.
    title : str
        Figure title.
    """
    if depth.ndim != 2:
        raise ValueError(f"`depth` must be (H,W), got shape {depth.shape}.")

    missing = [k for k in ("fx", "fy", "cx", "cy") if k not in cam_intrinsic]
    if missing:
        raise KeyError(f"`cam` is missing keys {missing}. Expected fx, fy, cx, cy.")

    fx = float(cam_intrinsic["fx"])
    fy = float(cam_intrinsic["fy"])
    cx = float(cam_intrinsic["cx"])
    cy = float(cam_intrinsic["cy"])

    scene_stride = int(cam_intrinsic.get("scene_stride", 6))
    z_min = float(cam_intrinsic.get("z_min", 0.15))
    z_max = float(cam_intrinsic.get("z_max", 2.0))
    max_points = cam_intrinsic.get("max_points", 120000)

    rgb_np: np.ndarray | None
    if rgb is None:
        rgb_np = None
    elif isinstance(rgb, np.ndarray):
        rgb_np = rgb
    elif hasattr(rgb, "convert"):
        rgb_np = np.asarray(rgb.convert("RGB"))
    else:
        raise TypeError("`rgb` must be a numpy array, a PIL image-like object, or None.")

    depth_m = depth.astype(np.float32, copy=False)
    if float(np.nanmax(depth_m)) > 100.0:
        depth_m = depth_m / 1000.0

    xyz, colors = _depth_to_pointcloud(
        depth_m, rgb_np, fx, fy, cx, cy,
        stride=int(scene_stride),
        z_min=float(z_min),
        z_max=float(z_max),
    )
    if xyz.shape[0] == 0:
        raise ValueError("No valid depth points for current (stride, z_min, z_max) settings.")

    if max_points is not None and xyz.shape[0] > int(max_points):
        sel = np.random.choice(xyz.shape[0], size=int(max_points), replace=False)
        xyz = xyz[sel]
        if colors is not None:
            colors = colors[sel]

    fig = go.Figure()

    if colors is not None:
        pc_color = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]
        marker = dict(size=1.5, color=pc_color, opacity=0.85)
    else:
        marker = dict(size=1.5, color="#888888", opacity=0.85)

    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=marker,
            name="Point Cloud",
            hoverinfo="skip",
        )
    )

    if r_curve is not None:
        fig.add_trace(
            go.Scatter3d(
                x=r_curve[:, 0],
                y=r_curve[:, 1],
                z=r_curve[:, 2],
                mode="lines",
                line=dict(width=6, color="blue"),
                name="Motion curve",
            )
        )

        if R_curve is not None:
            p_end = r_curve[-1]
            T_end = R_curve[-1, :, 0]
            # Adaptive arrow size: ~12% of curve arc length, clamped so very
            # short curves still show a visible cone and very long curves
            # don't overwhelm the scene.
            curve_len = float(np.linalg.norm(np.diff(r_curve, axis=0), axis=1).sum())
            arrow_len = float(np.clip(0.12 * curve_len, 0.015, 0.30))
            fig.add_trace(
                go.Cone(
                    x=[float(p_end[0])],
                    y=[float(p_end[1])],
                    z=[float(p_end[2])],
                    u=[float(T_end[0]) * arrow_len],
                    v=[float(T_end[1]) * arrow_len],
                    w=[float(T_end[2]) * arrow_len],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=1.2 * arrow_len,
                    colorscale=[[0, "blue"], [1, "blue"]],
                    showscale=False,
                    hoverinfo="skip",
                    name="Curve End Direction",
                    showlegend=False,
                )
            )

    # Three background presets, switchable in the rendered HTML via buttons.
    _BG_PRESETS = {
        "Light": dict(paper="white",   axis="rgb(245,245,245)", grid="rgb(220,220,220)", font="black"),
        "Dark":  dict(paper="#111111", axis="rgb(20,20,20)",    grid="rgb(60,60,60)",    font="white"),
        "Gray":  dict(paper="#444444", axis="rgb(70,70,70)",    grid="rgb(100,100,100)", font="white"),
    }

    def _bg_args(p):
        ax = dict(backgroundcolor=p["axis"], gridcolor=p["grid"], zerolinecolor=p["grid"],
                  showbackground=True, color=p["font"])
        return [{
            "paper_bgcolor": p["paper"],
            "plot_bgcolor":  p["paper"],
            "font.color":    p["font"],
            "scene.xaxis":   ax,
            "scene.yaxis":   ax,
            "scene.zaxis":   ax,
        }]

    default = _BG_PRESETS["Light"]

    fig.update_layout(
        width=1100,
        height=800,
        paper_bgcolor=default["paper"],
        plot_bgcolor=default["paper"],
        font=dict(color=default["font"]),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            dragmode="orbit",
            xaxis=dict(backgroundcolor=default["axis"], gridcolor=default["grid"],
                       zerolinecolor=default["grid"], showbackground=True),
            yaxis=dict(backgroundcolor=default["axis"], gridcolor=default["grid"],
                       zerolinecolor=default["grid"], showbackground=True),
            zaxis=dict(backgroundcolor=default["axis"], gridcolor=default["grid"],
                       zerolinecolor=default["grid"], showbackground=True),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title=title,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[dict(label=name, method="relayout", args=_bg_args(p))
                         for name, p in _BG_PRESETS.items()],
                direction="right",
                x=0.0, y=1.08, xanchor="left", yanchor="bottom",
                showactive=True,
            ),
        ],
    )
    return fig



# ----------------------------
# Example: helix-like curve
# ----------------------------
