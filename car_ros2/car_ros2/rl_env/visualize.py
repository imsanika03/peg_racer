import math
from typing import Callable, Optional
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from car_dynamics.controllers_jax.jax_waypoint import init_waypoints, generate
from rl_env.jit_neppo import load_path
import torch as th


from typing import Callable, Optional, Tuple


# --------- small drawing utils ---------
def _compute_track_edges(center_xy: np.ndarray, half_width_m: float) -> Tuple[np.ndarray, np.ndarray]:
    N = center_xy.shape[0]
    nxt = (np.arange(N) + 1) % N
    tang = center_xy[nxt] - center_xy
    norm = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    t_hat = tang / norm
    # +90° left normal
    n_left = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    left = center_xy + half_width_m * n_left
    right = center_xy - half_width_m * n_left
    return left, right

def _make_plot_axes(center_xy: np.ndarray, margin_m: float = 2.0):
    xmin, xmax = float(center_xy[:,0].min()), float(center_xy[:,0].max())
    ymin, ymax = float(center_xy[:,1].min()), float(center_xy[:,1].max())
    dx, dy = xmax - xmin, ymax - ymin
    cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
    half = 0.5*max(dx, dy) + margin_m

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor((0.07, 0.07, 0.07))
    fig.patch.set_facecolor((0.07, 0.07, 0.07))
    return fig, ax

def _draw_frame(ax, center_xy, left_xy, right_xy, cars_state):
    ax.plot(center_xy[:,0], center_xy[:,1], lw=1.5, alpha=0.85, color=(0.65,0.65,0.65))
    ax.plot(left_xy[:,0],   left_xy[:,1],   lw=1.0, alpha=0.7,  color=(0.35,0.35,0.35))
    ax.plot(right_xy[:,0],  right_xy[:,1],  lw=1.0, alpha=0.7,  color=(0.35,0.35,0.35))

    xs, ys, psis = cars_state
    colors = [(1.00, 0.35, 0.35), (0.35, 1.00, 0.60), (0.60, 0.60, 1.00)]
    for i in range(len(xs)):
        ax.scatter([xs[i]], [ys[i]], s=35, color=colors[i % 3], zorder=3)
        hx = xs[i] + 0.8 * math.cos(psis[i])
        hy = ys[i] + 0.8 * math.sin(psis[i])
        ax.plot([xs[i], hx], [ys[i], hy], lw=2, color=colors[i % 3], zorder=3)

# --------- main stateless renderer (policy-driven) ---------
def render_episode(
    reset_jit: Callable[[jax.Array], tuple],
    step_jit: Callable[[object, jnp.ndarray], tuple],
    policy: Callable[[jnp.ndarray, object, jax.Array], Tuple[np.ndarray, object]],
    policy_ctx: object,
    out_gif_path: str,
    *,
    track_half_width_m: float = 3.5,
    margin_m: float = 2.0,
    fps: int = 20,
    max_steps: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """
    Runs one episode (until done/truncated or max_steps) and saves a GIF.

    policy signature: action_np, policy_ctx = policy(obs_b: jnp.ndarray, policy_ctx, key)
      - obs_b: shape (3, obs_dim)
      - returns action for all 3 cars: shape (3, 2) in env action space
    """
    # Precompute track geometry from centerline
    path_yaml = "/Users/sanikabharvirkar/Documents/alpha-RACER/simulators/params-num.yaml"
    path = load_path(path_yaml)          # expects columns [s, x, y, ...]
    center_xy = np.asarray(path[:, 1:3], dtype=np.float32)
    left_xy, right_xy = _compute_track_edges(center_xy, half_width_m=track_half_width_m)

    # Reset env
    key = jax.random.PRNGKey(seed)
    key, sk = jax.random.split(key)
    state, obs0 = reset_jit(sk)
    obs = np.asarray(obs0, dtype=np.float32)

    frames = []
    step_count = 0

    while True:
        # ---- render frame ----
        cars = state.cars
        xs   = np.array(cars.x)
        ys   = np.array(cars.y)
        psis = np.array(cars.psi)

        fig, ax = _make_plot_axes(center_xy, margin_m)
        _draw_frame(ax, center_xy, left_xy, right_xy, (xs, ys, psis))
        ax.text(0.02, 0.98, f"t = {step_count}", color="w",
                transform=ax.transAxes, ha="left", va="top", fontsize=9)

        # Draw & capture (Agg-safe): use buffer_rgba() then drop alpha
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # HxWx4
        frame = rgba[..., :3].copy()  # RGB
        frames.append(frame)
        plt.close(fig)

        # ---- policy action ----
        obs_b = jnp.asarray(obs, dtype=jnp.float32)
        key, sk = jax.random.split(key)
        action_np, policy_ctx = policy(obs_b, policy_ctx, sk)  # user-provided policy
        action_j = jnp.asarray(action_np, dtype=jnp.float32)

        # ---- step env ----
        state, next_obs_j, reward_j, done_j, trunc_j, _ = step_jit(state, action_j)

        # unwrap step flags
        terminated = bool(np.asarray(done_j))
        truncated  = bool(np.asarray(trunc_j))

        step_count += 1
        if terminated or truncated:
            break
        if max_steps is not None and step_count >= max_steps:
            break

        obs = np.asarray(next_obs_j, dtype=np.float32)

    # Write GIF
    duration = 1.0 / max(1, fps)
    imageio.mimsave(out_gif_path, frames, duration=duration, loop=0)
    return {"frames": len(frames), "steps": step_count, "gif_path": out_gif_path}

def _compute_track_edges(center_xy: np.ndarray, half_width_m: float) -> Tuple[np.ndarray, np.ndarray]:
    N = center_xy.shape[0]
    nxt = (np.arange(N) + 1) % N
    tang = center_xy[nxt] - center_xy
    norm = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    t_hat = tang / norm
    # +90° left normal
    n_left = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    left = center_xy + half_width_m * n_left
    right = center_xy - half_width_m * n_left
    return left, right

def _make_plot_axes(center_xy: np.ndarray, margin_m: float = 2.0):
    xmin, xmax = float(center_xy[:,0].min()), float(center_xy[:,0].max())
    ymin, ymax = float(center_xy[:,1].min()), float(center_xy[:,1].max())
    dx, dy = xmax - xmin, ymax - ymin
    cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
    half = 0.5*max(dx, dy) + margin_m

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)  # Reduced DPI for memory efficiency
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor((0.07, 0.07, 0.07))
    fig.patch.set_facecolor((0.07, 0.07, 0.07))
    return fig, ax

def _draw_frame(ax, center_xy, left_xy, right_xy, cars_state):
    ax.plot(center_xy[:,0], center_xy[:,1], lw=1.5, alpha=0.85, color=(0.65,0.65,0.65))
    ax.plot(left_xy[:,0],   left_xy[:,1],   lw=1.0, alpha=0.7,  color=(0.35,0.35,0.35))
    ax.plot(right_xy[:,0],  right_xy[:,1],  lw=1.0, alpha=0.7,  color=(0.35,0.35,0.35))

    xs, ys, psis = cars_state
    colors = [(1.00, 0.35, 0.35), (0.35, 1.00, 0.60), (0.60, 0.60, 1.00)]
    for i in range(len(xs)):
        ax.scatter([xs[i]], [ys[i]], s=35, color=colors[i % 3], zorder=3)
        hx = xs[i] + 0.8 * math.cos(psis[i])
        hy = ys[i] + 0.8 * math.sin(psis[i])
        ax.plot([xs[i], hx], [ys[i], hy], lw=2, color=colors[i % 3], zorder=3)


def render_episode_to_tensor(
    reset_jit: Callable[[jax.Array], tuple],
    step_jit: Callable[[object, jnp.ndarray], tuple],
    policy: Callable[[jnp.ndarray, object, jax.Array], Tuple[np.ndarray, object]],
    policy_ctx: object,
    *,
    track_half_width_m: float = 3.5,
    margin_m: float = 2.0,
    max_steps: int = 50,
    seed: int = 0,
) -> th.Tensor:
    """
    Runs one episode and returns frames as a torch tensor for wandb logging.
    
    Returns:
        torch.Tensor: Shape (T, H, W, C) where T is number of frames
    """
    # Precompute track geometry from centerline
    path_yaml = "/Users/sanikabharvirkar/Documents/alpha-RACER/simulators/params-num.yaml"
    path = load_path(path_yaml)          # expects columns [s, x, y, ...]
    center_xy = np.asarray(path[:, 1:3], dtype=np.float32)
    left_xy, right_xy = _compute_track_edges(center_xy, half_width_m=track_half_width_m)

    # Reset env
    key = jax.random.PRNGKey(seed)
    key, sk = jax.random.split(key)
    state, obs0 = reset_jit(sk)
    obs = np.asarray(obs0, dtype=np.float32)

    frames = []
    step_count = 0

    while step_count < max_steps:
        # ---- render frame ----
        cars = state.cars
        xs   = np.array(cars.x)
        ys   = np.array(cars.y)
        psis = np.array(cars.psi)

        fig, ax = _make_plot_axes(center_xy, margin_m)
        _draw_frame(ax, center_xy, left_xy, right_xy, (xs, ys, psis))
        ax.text(0.02, 0.98, f"t = {step_count}", color="w",
                transform=ax.transAxes, ha="left", va="top", fontsize=9)

        # Draw & capture (Agg-safe): use buffer_rgba() then drop alpha
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # HxWx4
        frame = rgba[..., :3].copy()  # RGB
        frames.append(frame)
        plt.close(fig)

        # ---- policy action ----
        obs_b = jnp.asarray(obs, dtype=jnp.float32)
        key, sk = jax.random.split(key)
        action_np, policy_ctx = policy(obs_b, policy_ctx, sk)  # user-provided policy
        action_j = jnp.asarray(action_np, dtype=jnp.float32)

        # ---- step env ----
        state, next_obs_j, reward_j, done_j, trunc_j, _ = step_jit(state, action_j)

        # unwrap step flags
        terminated = bool(np.asarray(done_j))
        truncated  = bool(np.asarray(trunc_j))

        step_count += 1
        if terminated or truncated:
            break

        obs = np.asarray(next_obs_j, dtype=np.float32)

    # Convert frames list to numpy array then to torch tensor
    frames_array = np.stack(frames, axis=0)  # (T, H, W, C)
    frames_tensor = th.tensor(frames_array, dtype=th.uint8)
    
    return frames_tensor