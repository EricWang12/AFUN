"""Motion-curve math shared by the spline motion head, the robot dataset and inference.

Three groups of helpers:

* **Frenet-Serret curves** — ``fsparams_to_tensor`` / ``tensor_to_fsparams`` convert
  between the stored trajectory dict and the packed 8-D parameter vector;
  ``sample_fs_curve`` and ``compute_fsframe_curve`` integrate that parameterisation
  into 3-D positions. The robot dataset uses these to build the ``curve_samples``
  ground truth from ``trajectory.json``.
* **Bezier curves** — ``compute_spline_curve`` turns the motion head's predicted
  control points into sampled 3-D positions.
* **Visualisation** — ``plotly_pointcloud_and_curve`` renders a depth point cloud
  plus a motion curve as an interactive Plotly figure.
"""

import numpy as np
import plotly.graph_objects as go
import torch

# Defaults for the fixed-timestep Frenet-Serret integrator.
FPS = 30.0
FRAMES = 30


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


def _safe_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    
    return x / (torch.linalg.norm(x) + eps)


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

    ts = torch.linspace(0.0, 1.0, frames + 1, dtype=torch.float64, device=ctrl.device)

    def _bernstein_basis(tt: torch.Tensor, deg: int) -> torch.Tensor:
        # Return (deg+1, T) matrix of Bernstein weights with exact endpoint handling.
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


