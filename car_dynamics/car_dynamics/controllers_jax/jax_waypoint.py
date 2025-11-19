from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np



WaypointKind = Literal['circle','oval','square','custom']

@dataclass
class WaypointSpec:
    kind: WaypointKind
    dt: float
    H: int
    speed: float
    path: Optional[jnp.ndarray] = None
    scale: float = 1.0
    waypoint_t: Optional[jnp.ndarray] = None
    waypoint_xy: Optional[jnp.ndarray] = None

jax.tree_util.register_pytree_node(
    WaypointSpec,
    lambda s: ((s.path, s.waypoint_t, s.waypoint_xy),
               (s.kind, s.dt, s.H, s.speed, s.scale)),
    lambda data, aux: WaypointSpec(aux[0], aux[1], aux[2], aux[3], data[0], aux[4], data[1], data[2])
)

def counter_circle(theta: jnp.ndarray) -> jnp.ndarray:
    center = jnp.array([0.0, 0.0])
    radius = 4.5
    return center + radius * jnp.array([jnp.cos(theta), jnp.sin(theta)])

def counter_oval(theta: jnp.ndarray) -> jnp.ndarray:

    center = jnp.array([0.0, 0.0])
    x_radius = 1.2
    y_radius = 1.2, 1.4

    return center + jnp.array([x_radius*jnp.cos(theta), y_radius*jnp.sin(theta)])

def seg_len(x1,y1,x2,y2):
    return jnp.hypot(x2-x1, y2-y1)

def get_curvature(x1,y1, x2,y2, x3,y3):

    a = seg_len(x1,y1,x2,y2)
    b = seg_len(x2,y2,x3,y3)
    c = seg_len(x1,y1,x3,y3)
    s = 0.5*(a+b+c)

    area_term = jnp.maximum(s*(s-a)*(s-b)*(s-c), 0.0) #ts herrons
    num = 4.0*jnp.sqrt(area_term)
    den = jnp.maximum(a*b*c, 1e-12)
    prod = (x2-x1)*(y3-y2) - (x3-x2)*(y2-y1)
    k = (num/den) * jnp.sign(prod)
    return k


def counter_square(theta: jnp.ndarray) -> jnp.ndarray:

    center = jnp.array([1.0, 2.0])
    x_radius, y_radius = 4.0, 4.0
    r = jnp.sqrt(x_radius**2 + y_radius**2)
    th = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


    cond_right = (th >= -jnp.pi/4) & (th <= jnp.pi/4)
    cond_bottom = (th >= -3 * jnp.pi/4) & (th <= -jnp.pi/4)
    cond_left = ((th >= 3 * jnp.pi/4) & (th <= jnp.pi)) | ((th >= -jnp.pi) & (th <= -3*jnp.pi/4))

    x_right = center[0] + x_radius
    y_right = center[1] + r * jnp.sin(th)

    x_bottom = center[0] + r * jnp.cos(th)
    y_bottom = center[1] - y_radius

    x_left = center[0] - x_radius
    y_left = center[1] + r * jnp.sin(th)

    x_top = center[0] + r * jnp.cos(th)
    y_top = center[1] + y_radius

    x = jnp.where(cond_right, x_right,
                    jnp.where(cond_bottom, x_bottom,
                            jnp.where(cond_left, x_left, x_top)))
    
    y = jnp.where(cond_right, y_right,
                    jnp.where(cond_bottom, y_bottom,
                            jnp.where(cond_left, y_left, y_top)))

    return jnp.stack([x, y], axis=0)


def line_point_t(px, py, x1, y1, x2, y2):

    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    denom = jnp.maximum(denom, 1e-12)
    t = ((px - x1) * dx + (py - y1) * dy) / denom
    return t  

def clamp01(x):
    return jnp.minimum(1.0, jnp.maximum(0.0, x))


def custom_fn(theta: jnp.ndarray, traj: jnp.ndarray):

    t_end = traj[-1,0]
    theta_wrapped = jnp.mod(theta, t_end)

    i = jnp.clip(jnp.searchsorted(traj[:,0], theta_wrapped, side='right') - 1, 0, traj.shape[0]-2)

    t0 = traj[i, 0] 
    t1 = traj[i+1, 0]
    x0 = traj[i, 1]
    y0 = traj[i, 2]
    v0 = traj[i, 3]
    x1 = traj[i+1, 1]
    y1 = traj[i+1, 2]
    v1 = traj[i+1, 3]

    ratio = (theta_wrapped - t0) / jnp.maximum(t1 - t0, 1e-12)
    x = x0 + ratio * (x1 - x0)
    y = y0 + ratio * (y1 - y0)
    v = v0 + ratio * (v1 - v0)

    N = traj.shape[0]
    i1 = (i + 1) % N
    i2 = (i + 2) % N

    k = get_curvature(traj[i,1], traj[i,2], traj[i1,1], traj[i1,2], traj[i2,1], traj[i2,2])

    return jnp.stack([x,y], axis=0), v, {'curv': k}

def calc_shifted_traj(traj_xy: jnp.ndarray, shift_dist: float) -> jnp.ndarray:

    prev = jnp.roll(traj_xy,  1, axis=0)
    nxt  = jnp.roll(traj_xy, -1, axis=0)

    yaws = jnp.arctan2(nxt[:,1] - prev[:,1], nxt[:,0] - prev[:,0])

    nx = jnp.cos(yaws + jnp.pi/2.0)
    ny = jnp.sin(yaws + jnp.pi/2.0)
    return traj_xy + shift_dist * jnp.stack([nx,ny], axis=1)


def waypoint_shape(kind: WaypointKind):
    if kind == 'circle':
        return counter_circle
    if kind == 'oval':
        return counter_oval
    if kind == 'square':
        return counter_square
    raise ValueError("No analytic pos fn for custom.")

def init_waypoints(kind: WaypointKind, dt: float, H: int, speed: float,
                   path: Optional[jnp.ndarray]=None, scale: float=1.0) -> WaypointSpec:

    if kind == 'custom':
        assert path is not None, "For kind='custom', you must pass path (N,4)."
        t_max = path[-1,0]
        waypoint_t = jnp.arange(0.0, t_max, 0.1 * scale)

        def f(t):
            pos, _, _ = custom_fn(t, path)
            return pos
        waypoint_xy = jax.vmap(f)(waypoint_t) 
        return WaypointSpec(kind, dt, H, speed, path=path, scale=scale,
                            waypoint_t=waypoint_t, waypoint_xy=waypoint_xy)
    else:
        pos_fn = waypoint_shape(kind)
        waypoint_t = jnp.arange(0.0, 2.0*jnp.pi + dt, dt / jnp.maximum(speed, 1e-6))
        waypoint_xy = jax.vmap(pos_fn)(waypoint_t) 
        return WaypointSpec(kind, dt, H, speed, path=None, scale=1.0,
                            waypoint_t=waypoint_t, waypoint_xy=waypoint_xy)


def _nearest_waypoint_idx(waypoint_xy: jnp.ndarray, pos2d: jnp.ndarray):

    dists = jnp.linalg.norm(waypoint_xy - pos2d[None, :], axis=-1)

    return jnp.argmin(dists)

def _refine_along_segments(waypoint_t: jnp.ndarray, waypoint_xy: jnp.ndarray, i: jnp.ndarray, pos2d: jnp.ndarray):

    N = waypoint_xy.shape[0]
    im1 = (i - 1) % N
    ip1 = (i + 1) % N

    Pi = waypoint_xy[i]
    Pim1 = waypoint_xy[im1]
    Pip1 = waypoint_xy[ip1]


    d_prev = seg_len(Pim1[0],Pim1[1], Pi[0],Pi[1])
    t1 = line_point_t(pos2d[0],pos2d[1], Pim1[0],Pim1[1], Pi[0],Pi[1])

    side1 = jnp.sign((Pi[0]-Pim1[0])*(pos2d[1]-Pim1[1]) - (Pi[1]-Pim1[1])*(pos2d[0]-Pim1[0]))
    pt1 = Pim1 + t1*(Pi - Pim1)
    dist1 = jnp.linalg.norm(pos2d - pt1) * side1
    d1 = d_prev*(clamp01(t1) - 1.0) 


    d_next = seg_len(Pi[0],Pi[1], Pip1[0],Pip1[1])
    t2 = line_point_t(pos2d[0],pos2d[1], Pi[0],Pi[1], Pip1[0],Pip1[1])
    side2 = jnp.sign((Pip1[0]-Pi[0])*(pos2d[1]-Pi[1]) - (Pip1[1]-Pi[1])*(pos2d[0]-Pi[0]))
    pt2 = Pi + t2*(Pip1 - Pi)
    dist2 = jnp.linalg.norm(pos2d - pt2) * side2
    d2 = d_next*clamp01(t2)

    final_dist = jnp.where(jnp.abs(dist1) < jnp.abs(dist2), dist1, dist2)

    return waypoint_t[i] + d1 + d2, final_dist

def gen_heading_and_speed(pos_fn, t, dt_eff):

    p0 = pos_fn(t)
    p1 = pos_fn(t + dt_eff)
    vel = (p1 - p0) / jnp.maximum(dt_eff, 1e-12)
    speed_ref = jnp.clip(jnp.linalg.norm(vel), 0.5, 100.0)
    psi = jnp.arctan2(vel[1], vel[0])
    return p0, speed_ref, psi, p1

def gen_analytic_targets(kind_fn, t0, H, dt, body_speed, speed_ref_scale):

    idxs = jnp.arange(H+1)
    def per_i(i):
        t = t0 + i * dt * body_speed
        p0, speed_ref, psi, _ = gen_heading_and_speed(kind_fn, t, dt*speed_ref_scale)
        return jnp.array([p0[0], p0[1], psi, speed_ref])
    return jax.vmap(per_i)(idxs)

def gen_custom_targets(path, t0, H, dt, mu_factor, body_speed):

    idxs = jnp.arange(H+1)

    def per_i(carry, i):
        t = t0 + i*dt*body_speed
        pos, speed, info = custom_fn(t, path)
        speed = speed * jnp.sqrt(mu_factor)
        pos_next, _, _ = custom_fn(t + dt*speed, path)
        vel = (pos_next - pos) / jnp.maximum(dt, 1e-12)
        speed_ref = jnp.clip(jnp.linalg.norm(vel), 0.5, 100.0)
        psi = jnp.arctan2(vel[1], vel[0])
        tgt = jnp.array([pos[0], pos[1], psi, info['curv'], speed])
        return carry, tgt

    _, out = lax.scan(per_i, None, idxs)
    return out 

def generate(spec: WaypointSpec,
             obs: jnp.ndarray,
             dt: float = -1.0,
             mu_factor: float = 1.0,
             body_speed: float = 1.0
            ):

    dt_use = jnp.where(dt < 0, spec.dt, dt)
    pos2d = obs[:2]

    if spec.kind != 'custom':

        i = _nearest_waypoint_idx(spec.waypoint_xy, pos2d)
        t_closed = spec.waypoint_t[i]

        t_refined, _ = _refine_along_segments(spec.waypoint_t, spec.waypoint_xy, i, pos2d)

        pos_fn = waypoint_shape(spec.kind)
        magic = 1.0/1.2
        targets = gen_analytic_targets(pos_fn, t_refined, spec.H, dt_use, spec.speed*magic, spec.speed*magic)
        return targets, None, None, None


    i = _nearest_waypoint_idx(spec.waypoint_xy, pos2d)
    t_closed = spec.waypoint_t[i]
    t_refined, final_dist = _refine_along_segments(spec.waypoint_t, spec.waypoint_xy, i, pos2d)

    pos0, speed0, _ = custom_fn(t_refined, spec.path)
    speed0 = speed0 * jnp.sqrt(mu_factor)
    kin_pos, _, _ = custom_fn(t_refined + 1.2*speed0, spec.path)

    targets = gen_custom_targets(spec.path, t_refined, spec.H, dt_use, mu_factor, body_speed)
    return targets, kin_pos, t_refined, final_dist

class JaxWaypointGenerator:

    def __init__(self, 
                 trajectory: str, 
                 dt: float, 
                 H: int, 
                 speed: float,
                 path: Optional[np.ndarray] = None, 
                 scale: float = 1.0):

        tr = trajectory.lower()

        if tr in ("counter circle", "circle"):
            self.kind = "circle"
            self.spec = init_waypoints(kind="circle", dt=dt, H=H, speed=speed)
            self.track_L = float(2.0*np.pi)

        elif tr in ("counter oval", "oval"):
            self.kind = "oval"
            self.spec = init_waypoints(kind="oval", dt=dt, H=H, speed=speed)
            self.track_L = float(2.0*np.pi)

        elif tr in ("counter square", "square"):
            self.kind = "square"
            self.spec = init_waypoints(kind="square", dt=dt, H=H, speed=speed)
            self.track_L = float(2.0*np.pi)

        else:

            if path is None:
                raise ValueError(
                    " `path` must be an (N,4) numpy array [t,x,y,v]."
                )
            jpath = jnp.array(path) 
            self.kind = "custom"
            self.spec = init_waypoints(kind="custom", dt=dt, H=H, speed=speed, path=jpath, scale=scale)
            self.track_L = float(path[-1,0])

        self._gen = jax.jit(generate, static_argnums=(0,))

    def generate(self, obs_5: jnp.ndarray, dt: float, mu_factor: float, body_speed: float):

        targets, kin_pos, s, e = self._gen(self.spec, obs_5, dt=dt, mu_factor=mu_factor, body_speed=body_speed)

        if self.kind != "custom":

            zeros = jnp.zeros((targets.shape[0], 1))
            targets = jnp.concatenate([targets[:, :3], zeros, targets[:, 3:4]], axis=1)
            
            s = s if s is not None else 0.0
            e = e if e is not None else 0.0

        return targets, kin_pos, s, e