from dataclasses import dataclass
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pandas as pd
import yaml
from car_dynamics.controllers_jax.jax_waypoint import init_waypoints, generate

GRAVITY = 9.81

class CarBatchState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    psi: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    omega: jnp.ndarray

class EnvState(NamedTuple):
    cars: CarBatchState
    delay_buf: jnp.ndarray
    t: jnp.int32
    last_rel: jnp.ndarray
    track_L: float

@dataclass
class DynamicParams:
    num_envs: int
    LF: float = .11
    LR: float = .23
    MASS: float = 4.65
    DT: float = .05
    K_RFY: float = 20.
    K_FFY: float = 20.
    Iz: float = 0.1
    Ta: float = 5.5
    Tb: float = -1.
    Sa: float = 0.36
    Sb: float = 0.03
    mu: float = 3.0
    Cf: float = 1.0
    Cr: float = 1.0
    Bf: float = 60.0
    Br: float = 60.0
    hcom: float = 0.0
    fr: float = 0.1
    delay: int = 1

# ---------------- Dynamics ----------------

def dbm_dxdt(
    x, y, psi, vx, vy, omega, target_vel, target_steer,
    Ta,Tb,Sa,Sb, LF,LR, MASS, K_RFY,K_FFY, Iz, mu, Cf,Cr,Bf,Br, hcom, fr
):
    steer = target_steer * Sa + Sb
    prev_vel = jnp.hypot(vx, vy)
    throttle = target_vel * Ta - target_vel * Tb * prev_vel

    next_x   = (vx * jnp.cos(psi) - vy * jnp.sin(psi))
    next_y   = (vx * jnp.sin(psi) + vy * jnp.cos(psi))
    next_psi = omega

    alpha_f = steer - jnp.arctan((LF * omega + vy) / jnp.maximum(vx, 0.5))
    alpha_r = jnp.arctan((LR * omega - vy) / jnp.maximum(vx, 0.5))

    F_rx = throttle - fr * MASS * GRAVITY * jnp.sign(vx)

    F_fz = 0.5 * MASS * GRAVITY * LR / (LF + LR) - 0.5 * hcom / (LF + LR) * F_rx
    F_rz = 0.5 * MASS * GRAVITY * LF / (LF + LR) + 0.5 * hcom / (LF + LR) * F_rx

    F_fy = 2 * mu * F_fz * jnp.sin(Cf * jnp.arctan(Bf * alpha_f))
    F_ry = 2 * mu * F_rz * jnp.sin(Cr * jnp.arctan(Br * alpha_r))

    ax = (F_rx - F_fy * jnp.sin(steer) + vy * omega * MASS) / MASS
    ay = (F_ry + F_fy * jnp.cos(steer) - vx * omega * MASS) / MASS
    adot = (F_fy * LF * jnp.cos(steer) - F_ry * LR) / Iz
    return next_x, next_y, next_psi, ax, ay, adot

def rk4_step(params: DynamicParams, state, target_vel, target_steer):
    DT = params.DT
    K1 = dbm_dxdt(*state, target_vel, target_steer,
                  params.Ta, params.Tb, params.Sa, params.Sb, params.LF, params.LR,
                  params.MASS, params.K_RFY, params.K_FFY, params.Iz, params.mu,
                  params.Cf, params.Cr, params.Bf, params.Br, params.hcom, params.fr)

    S2 = tuple(state[i] + 0.5*DT*K1[i] for i in range(6))
    K2 = dbm_dxdt(*S2, target_vel, target_steer,
                  params.Ta, params.Tb, params.Sa, params.Sb, params.LF, params.LR,
                  params.MASS, params.K_RFY, params.K_FFY, params.Iz, params.mu,
                  params.Cf, params.Cr, params.Bf, params.Br, params.hcom, params.fr)

    S3 = tuple(state[i] + 0.5*DT*K2[i] for i in range(6))
    K3 = dbm_dxdt(*S3, target_vel, target_steer,
                  params.Ta, params.Tb, params.Sa, params.Sb, params.LF, params.LR,
                  params.MASS, params.K_RFY, params.K_FFY, params.Iz, params.mu,
                  params.Cf, params.Cr, params.Bf, params.Br, params.hcom, params.fr)

    S4 = tuple(state[i] + DT*K3[i] for i in range(6))
    K4 = dbm_dxdt(*S4, target_vel, target_steer,
                  params.Ta, params.Tb, params.Sa, params.Sb, params.LF, params.LR,
                  params.MASS, params.K_RFY, params.K_FFY, params.Iz, params.mu,
                  params.Cf, params.Cr, params.Bf, params.Br, params.hcom, params.fr)

    nx = state[0] + DT/6.0 * (K1[0] + 2*K2[0] + 2*K3[0] + K4[0])
    ny = state[1] + DT/6.0 * (K1[1] + 2*K2[1] + 2*K3[1] + K4[1])
    npsi = state[2] + DT/6.0 * (K1[2] + 2*K2[2] + 2*K3[2] + K4[2])
    nvx = state[3] + DT/6.0 * (K1[3] + 2*K2[3] + 2*K3[3] + K4[3])
    nvy = state[4] + DT/6.0 * (K1[4] + 2*K2[4] + 2*K3[4] + K4[4])
    nomega = state[5] + DT/6.0 * (K1[5] + 2*K2[5] + 2*K3[5] + K4[5])
    return (nx, ny, npsi, nvx, nvy, nomega)

# ---------------- Safety filter (geometry-safe backup + hard zone) ----------------
def safety_filter(a_raw, feats, lane_half_width_m, psi_ref, psi_body, prev_action=None):
    """
    a_raw   : (3,2) [vel, steer] in env scale
    feats   : (3,8) [s, e, theta_diff, vx, vy, omega, curv, curv_lh]
    psi_ref : (3,)   reference heading from waypoints
    psi_body: (3,)   current body yaw
    """
    e        = feats[:, 1]                   # signed lateral error (meters)
    vx       = jnp.maximum(feats[:, 3], 0.)  # forward speed estimate
    curv     = feats[:, 6]

    # --- baseline caps (curve-aware) ---
    v_cap_curve = 0.9 / (1.0 + 6.0 * jnp.abs(curv))

    # --- hard-zone geometry ---
    margin     = 0.50  # start hard safety this far before the edge
    safe_band  = jnp.maximum(lane_half_width_m - margin, 1e-3)
    in_hard    = (jnp.abs(e) >= safe_band)
    outside    = (jnp.abs(e) > lane_half_width_m)

    # geometric steer: head error wrt centerline + cross-track term
    # NOTE: use -e if your e is "positive to left"; this pulls back to centerline.
    FLIP_E   = True
    e_fix    = jnp.where(FLIP_E, -e, e)
    hdg_err  = jnp.arctan2(jnp.sin(psi_ref - psi_body), jnp.cos(psi_ref - psi_body))
    k_th_soft, k_e_soft = 1.0, 1.2
    k_th_hard, k_e_hard = 1.6, 3.0
    eps = 0.2

    steer_soft = k_th_soft*hdg_err + jnp.arctan2(k_e_soft*e_fix, vx + eps)
    steer_hard = k_th_hard*hdg_err + jnp.arctan2(k_e_hard*e_fix, vx + eps)

    # raw clipped
    vel_raw   = jnp.clip(a_raw[:, 0], 0.0, 1.0)
    steer_raw = jnp.clip(a_raw[:, 1], -1.0, 1.0)

    # small steering-rate limit (only on raw path, not on hard overwrite)
    if prev_action is not None:
        max_rate = 0.15
        steer_raw = jnp.clip(steer_raw,
                             prev_action[:,1] - max_rate,
                             prev_action[:,1] + max_rate)

    # choose steer by zone
    steer_safe = jnp.where(in_hard, steer_hard, steer_soft)
    steer_safe = jnp.clip(steer_safe, -1.0, 1.0)

    # speed caps by zone
    v_cap = jnp.minimum(v_cap_curve, jnp.where(in_hard, 0.30, 0.90))
    v_cap = jnp.where(outside, 0.05, v_cap)   # crawl if already outside

    vel_safe = jnp.clip(vel_raw, 0.0, v_cap)

    a_safe = jnp.stack([vel_safe, steer_safe], axis=1)
    return a_safe, in_hard, outside



# ---------------- Helpers ----------------

def wrap_diff(a, b, L):
    d = a - b
    d = jnp.where(d < -L/2., d + L, d)
    d = jnp.where(d >  L/2., d - L, d)
    return d

def angle_diff(a, b):
    return jnp.arctan2(jnp.sin(a-b), jnp.cos(a-b))

# ---------------- Env build ----------------

def build_old_env_functions(params: DynamicParams,
                        EP_LEN: int,
                        track_L: float,
                        delay: int,
                        wp_generate,
                        lane_half_width_m: float = 3.5,
                        vmax_mps: float = 6.0):
    """Residual reference control + safety; sparse reward = absolute track progress only."""

    def spawn_poses():
        return jnp.array([
            [ 3.0,  5.0, -jnp.pi/2 - 0.72],
            [ 0.0,  0.0, -jnp.pi/2 - 0.50],
            [-2.0, -6.0, -jnp.pi/2 - 0.50],
        ], dtype=jnp.float32)

    # features (unchanged layout)
    def feats_from(cars):
        def one(i):
            obs5 = jnp.array([cars.x[i], cars.y[i], cars.psi[i], cars.vx[i], cars.vy[i]])
            tgt, _, s, e = wp_generate(obs5, cars.vx[i])  # targets: [x,y,psi,curv,speed]
            theta = tgt[0,2]
            theta_diff = angle_diff(theta, cars.psi[i])
            curv = tgt[0,3]
            curv_lh = tgt[-1,3]
            return jnp.array([s, e, theta_diff, cars.vx[i], cars.vy[i], cars.omega[i], curv, curv_lh], jnp.float32)
        return jax.vmap(one)(jnp.arange(3))

    # reference heading & speed for residual control
    def refs_from(cars):
        def one(i):
            obs5 = jnp.array([cars.x[i], cars.y[i], cars.psi[i], cars.vx[i], cars.vy[i]])
            tgt, _, _, _ = wp_generate(obs5, cars.vx[i])
            psi_ref   = tgt[0,2]
            speed_ref = tgt[0,4]  # from custom path file
            return jnp.array([psi_ref, speed_ref], jnp.float32)
        return jax.vmap(one)(jnp.arange(3))  # (3,2)

    # normalized RL obs
    def _norm_obs_from_feats(feats, self_i):
        a = (self_i + 1) % 3
        b = (self_i + 2) % 3
        da = jnp.abs(wrap_diff(feats[a,0], feats[self_i,0], track_L))
        db = jnp.abs(wrap_diff(feats[b,0], feats[self_i,0], track_L))
        front_idx = jnp.where(da <= db, a, b)

        front = jnp.take(feats, front_idx, axis=0)
        fself = jnp.take(feats, self_i,   axis=0)

        def ang(x): return x / jnp.pi
        def spd(x): return jnp.clip(x / vmax_mps, -1., 1.)
        def lat(x): return jnp.clip(x / lane_half_width_m, -2., 2.)

        return jnp.array([
            wrap_diff(front[0], fself[0], track_L) / track_L,  # relative s (0..1)
            lat(front[1]),  lat(fself[1]),
            ang(front[2]),
            spd(front[3]), spd(front[4]), fself[5],            # omega left unscaled
            ang(fself[2]),
            spd(fself[3]), spd(fself[4]), fself[5],
            front[6],  fself[6],
            front[7],  fself[7],
        ], dtype=jnp.float32)

    def jax_reset(key: jax.Array) -> Tuple[EnvState, jnp.ndarray]:
        poses = spawn_poses()
        cars = CarBatchState(
            x=poses[:,0], y=poses[:,1], psi=poses[:,2],
            vx=jnp.zeros(3), vy=jnp.zeros(3), omega=jnp.zeros(3),
        )
        delay_buf = jnp.zeros((3, delay, 2), dtype=jnp.float32)

        # initial feats
        def car_features(i, _):
            obs5 = jnp.array([cars.x[i], cars.y[i], cars.psi[i], cars.vx[i], cars.vy[i]])
            tgt, _, s, e = wp_generate(obs5, cars.vx[i])
            theta = tgt[0,2]
            theta_diff = angle_diff(theta, cars.psi[i])
            curv = tgt[0,3]
            curv_lh = tgt[-1,3]
            return 0, jnp.array([s, e, theta_diff, cars.vx[i], cars.vy[i], cars.omega[i], curv, curv_lh], dtype=jnp.float32)

        _, feats0 = lax.scan(car_features, 0, jnp.arange(3))

        def rel_for(i):
            a = (i + 1) % 3
            b = (i + 2) % 3
            s_self = feats0[i, 0]
            s_a = feats0[a, 0]
            s_b = feats0[b, 0]
            return wrap_diff(s_self, jnp.maximum(s_a, s_b), track_L)

        last_rel = jax.vmap(rel_for)(jnp.arange(3))

        state = EnvState(cars=cars,
                         delay_buf=delay_buf,
                         t=jnp.array(0, jnp.int32),
                         last_rel=last_rel,
                         track_L=jnp.asarray(track_L, jnp.float32))

        obs0 = jax.vmap(lambda i: _norm_obs_from_feats(feats0, i))(jnp.arange(3))
        return state, obs0

    def jax_step(state: EnvState, action: jnp.ndarray):
        action = jnp.clip(action, -1.0, 1.0)

        feats_b   = feats_from(state.cars)
        refs_b    = refs_from(state.cars)
        psi_ref   = refs_b[:,0]
        speed_ref = jnp.clip(refs_b[:,1], 0.0, 1.0)

        hdg_err   = angle_diff(psi_ref, state.cars.psi)
        e         = feats_b[:,1]
        vx_b      = jnp.maximum(feats_b[:,3], 0.0)
        steer_ref = 1.2*hdg_err + jnp.arctan2(0.9*(-e), vx_b + 0.2)
        steer_ref = jnp.clip(steer_ref, -1.0, 1.0)

        res_vel   = action[:,0] * 0.30
        res_steer = action[:,1] * 0.40
        vel_cmd   = jnp.clip(speed_ref + res_vel, 0.0, 1.0)
        steer_cmd = jnp.clip(steer_ref + res_steer, -1.0, 1.0)
        a_for_filter = jnp.stack([vel_cmd, steer_cmd], axis=1)

        # ----- Strong safety -----
        prev_cmd = state.delay_buf[:,-1,:] if state.delay_buf.shape[1] > 0 else None
        a_safe, in_hard, outside = safety_filter(
            a_for_filter, feats_b, lane_half_width_m=lane_half_width_m,
            psi_ref=psi_ref, psi_body=state.cars.psi, prev_action=prev_cmd
        )

        # commands that *would* go through the buffer
        a0 = jnp.stack([jnp.clip(a_safe[:,0], 0., 1.),
                        jnp.clip(a_safe[:,1], -1., 1.)], axis=1)

        # ----- Delay buffer as usual -----
        if state.delay_buf.shape[1] > 0:
            buf1 = jnp.concatenate([a0[:,None,:], state.delay_buf[:,:-1,:]], axis=1)
            cmd_prev = buf1[:,-1,:]
            # **Bypass latency in the hard zone**: immediately apply safe action
            cmd = jnp.where(in_hard[:,None], a0, cmd_prev)
        else:
            buf1 = state.delay_buf
            cmd  = a0

        target_vel, target_steer = cmd[:,0], cmd[:,1]


        # integrate
        S = state.cars
        nx, ny, npsi, nvx, nvy, nomega = rk4_step(
            params, (S.x, S.y, S.psi, S.vx, S.vy, S.omega),
            target_vel, target_steer
        )

        cars2 = CarBatchState(x=nx, y=ny, psi=npsi, vx=nvx, vy=nvy, omega=nomega)

        # build normalized obs
        feats_before = feats_b
        feats_after  = feats_from(cars2)

        obs_before = jax.vmap(lambda i: _norm_obs_from_feats(feats_before, i))(jnp.arange(3))
        next_obs   = jax.vmap(lambda i: _norm_obs_from_feats(feats_after,  i))(jnp.arange(3))

        # ----- Sparse reward: absolute progress only -----
        s_before = feats_before[:, 0]
        s_after  = feats_after[:, 0]
        abs_prog = wrap_diff(s_after, s_before, track_L)
        abs_prog = jnp.maximum(abs_prog, 0.0)
        r = abs_prog

        t2 = state.t + jnp.int32(1)
        done = t2 >= jnp.int32(EP_LEN)
        truncated = done

        # keep last_rel for optional logging
        def rel_for(feats, i):
            aidx = (i + 1) % 3
            bidx = (i + 2) % 3
            s_self = feats[i,0]
            s_a = feats[aidx,0]
            s_b = feats[bidx,0]
            return wrap_diff(s_self, jnp.maximum(s_a, s_b), track_L)
        rel_after = jax.vmap(lambda i: rel_for(feats_after, i))(jnp.arange(3))

        state2 = EnvState(cars=cars2,
                          delay_buf=buf1,
                          t=t2,
                          last_rel=rel_after,
                          track_L=track_L)

        info_obs_before = obs_before
        return state2, next_obs, r, done, truncated, info_obs_before

    return jax_reset, jax_step

# ---------------- Path loading & builders ----------------

def load_path(waypoint_type):
    yaml_content = yaml.load(open(waypoint_type, 'r'), Loader=yaml.FullLoader)
    centerline_file = yaml_content['track_info']['centerline_file'][:-4]
    ox = yaml_content['track_info']['ox']
    oy = yaml_content['track_info']['oy']
    df = pd.read_csv('/Users/sanikabharvirkar/Documents/alpha-RACER/ref_trajs/' + centerline_file + '_with_speeds.csv')
    if waypoint_type.find('num') != -1:
        return np.array(df.iloc[:-1,:])*yaml_content['track_info']['scale'] + np.array([0, ox, oy, 0])
    else:
        return np.array(df.iloc[:,:]) + np.array([0, ox, oy, 0])

EP_LEN = 500

def build_old_step_and_reset(num_envs):
    params = DynamicParams(num_envs=num_envs, DT=0.1, Sa=0.34, Sb=0.0, Ta=20., Tb=0., mu=0.5, delay=4)

    path_rn = "/Users/sanikabharvirkar/Documents/alpha-RACER/simulators/params-num.yaml"
    path = load_path(path_rn)
    spec = init_waypoints(kind='custom', dt=0.1, H=9, speed=1.0, path=jnp.array(path), scale=6.5)
    track_L = float(path[-1,0])

    def wp_generate(obs5, vx):
        targets, kin_pos, s, e = generate(spec, obs5, dt=0.1, mu_factor=1.0, body_speed=vx)
        if targets.shape[1] == 4:
            zeros = jnp.zeros((targets.shape[0],1))
            targets = jnp.concatenate([targets[:,:3], zeros, targets[:,3:4]], axis=1)
        return targets, kin_pos, s, e

    reset_fn, step_fn = build_old_env_functions(params, EP_LEN, float(track_L), params.delay, wp_generate)
    reset_jit = jax.jit(reset_fn)
    step_jit  = jax.jit(step_fn, donate_argnums=(0,))
    return reset_jit, step_jit


def build_env_functions(
    params: DynamicParams,
    EP_LEN: int,
    track_L: float,
    delay: int,
    wp_generate,
    lane_half_width_m: float = 3.5,
    vmax_mps: float = 6.0,
    training_mode: bool = True
):
    """Improved environment with better reward shaping and training-friendly features."""
    
    def spawn_poses():
        """Initial spawn positions for 3 cars."""
        return jnp.array([
            [ 3.0,  5.0, -jnp.pi/2 - 0.72],
            [ 0.0,  0.0, -jnp.pi/2 - 0.50],
            [-2.0, -6.0, -jnp.pi/2 - 0.50],
        ], dtype=jnp.float32)
    
    def compute_features(cars):
        """Compute features for each car."""
        def one(i):
            obs5 = jnp.array([cars.x[i], cars.y[i], cars.psi[i], cars.vx[i], cars.vy[i]])
            tgt, _, s, e = wp_generate(obs5, cars.vx[i])
            theta = tgt[0,2]
            theta_diff = angle_diff(theta, cars.psi[i])
            curv = tgt[0,3]
            curv_lh = tgt[-1,3]
            return jnp.array([s, e, theta_diff, cars.vx[i], cars.vy[i], 
                            cars.omega[i], curv, curv_lh], jnp.float32)
        return jax.vmap(one)(jnp.arange(3))
    
    def get_references(cars):
        """Get reference heading and speed for each car."""
        def one(i):
            obs5 = jnp.array([cars.x[i], cars.y[i], cars.psi[i], cars.vx[i], cars.vy[i]])
            tgt, _, _, _ = wp_generate(obs5, cars.vx[i])
            psi_ref = tgt[0,2]
            speed_ref = tgt[0,4]
            return jnp.array([psi_ref, speed_ref], jnp.float32)
        return jax.vmap(one)(jnp.arange(3))
    
    def improved_safety_filter(a_raw, feats, lane_half_width_m, psi_ref, psi_body, 
                              prev_action=None):
        """
        Improved safety filter that's less restrictive during training.
        """
        e = feats[:, 1]  # lateral error
        vx = jnp.maximum(feats[:, 3], 0.0)
        curv = feats[:, 6]
        
        # Adaptive speed cap based on curvature
        v_cap_curve = 0.95 / (1.0 + 4.0 * jnp.abs(curv))  # Less aggressive reduction
        
        # Define safety zones with softer boundaries
        margin = 0.40  # Reduced from 0.50
        safe_band = jnp.maximum(lane_half_width_m - margin, 1e-3)
        in_hard = (jnp.abs(e) >= safe_band)
        outside = (jnp.abs(e) > lane_half_width_m)
        
        # Compute geometric steering
        e_fix = -e  # Pull toward centerline
        hdg_err = jnp.arctan2(jnp.sin(psi_ref - psi_body), jnp.cos(psi_ref - psi_body))
        
        # Softer steering parameters
        k_th_soft, k_e_soft = 0.8, 1.0
        k_th_hard, k_e_hard = 1.4, 2.5
        eps = 0.2
        
        steer_soft = k_th_soft * hdg_err + jnp.arctan2(k_e_soft * e_fix, vx + eps)
        steer_hard = k_th_hard * hdg_err + jnp.arctan2(k_e_hard * e_fix, vx + eps)
        
        # Raw action clipping
        vel_raw = jnp.clip(a_raw[:, 0], 0.0, 1.0)
        steer_raw = jnp.clip(a_raw[:, 1], -1.0, 1.0)
        
        # Steering rate limit (softer)
        if prev_action is not None:
            max_rate = 0.20  # Increased from 0.15
            steer_raw = jnp.clip(
                steer_raw,
                prev_action[:,1] - max_rate,
                prev_action[:,1] + max_rate
            )
        
        # Choose steering based on zone
        steer_safe = jnp.where(in_hard, steer_hard, steer_soft)
        steer_safe = jnp.clip(steer_safe, -1.0, 1.0)
        
        # Less restrictive speed caps for training
        if training_mode:
            v_cap = jnp.minimum(v_cap_curve, jnp.where(in_hard, 0.50, 0.95))
            v_cap = jnp.where(outside, 0.25, v_cap)
        else:
            v_cap = jnp.minimum(v_cap_curve, jnp.where(in_hard, 0.30, 0.90))
            v_cap = jnp.where(outside, 0.10, v_cap)
        
        vel_safe = jnp.clip(vel_raw, 0.0, v_cap)
        
        a_safe = jnp.stack([vel_safe, steer_safe], axis=1)
        return a_safe, in_hard, outside
    
    def compute_improved_reward(
        s_before, s_after, e_before, e_after, 
        vx_before, vx_after, in_hard, outside,
        rel_pos_before, rel_pos_after,
        track_L, lane_half_width_m
    ):
        """
        Comprehensive reward function with multiple components.
        
        Returns:
            reward: (3,) array of rewards for each car
            info: dict with reward component breakdowns
        """
        # 1. Progress reward (main objective)
        abs_prog = wrap_diff(s_after, s_before, track_L)
        progress_reward = jnp.maximum(abs_prog, 0.0) * 5.0  # Scaled progress
        
        # 2. Speed maintenance reward (encourage high speed)
        speed_reward = jnp.where(
            vx_after > 0.1,  # Only reward when moving
            (vx_after - 0.1) * 0.5,  # Proportional to speed
            0.0
        )
        
        # 3. Lane keeping reward (stay near centerline)
        # Smooth penalty that increases with distance from center
        centerline_reward = -jnp.power(jnp.abs(e_after) / lane_half_width_m, 2) * 0.3
        
        # 4. Improvement bonus (reward for getting closer to centerline)
        improvement = jnp.abs(e_before) - jnp.abs(e_after)
        improvement_bonus = jnp.where(improvement > 0, improvement * 0.2, 0.0)
        
        # 5. Safety penalties
        hard_zone_penalty = jnp.where(in_hard, -0.2, 0.0)
        outside_penalty = jnp.where(outside, -2.0, 0.0)
        
        # 6. Competitive racing rewards
        # Reward for improving relative position
        def compute_racing_reward(i):
            # Check if we overtook someone
            rel_before = rel_pos_before[i]
            rel_after = rel_pos_after[i]
            
            # Reward for gaining position
            position_gain = (rel_after - rel_before) / track_L
            racing_reward = jnp.clip(position_gain * 2.0, -0.5, 1.0)
            
            return racing_reward
        
        racing_rewards = jax.vmap(compute_racing_reward)(jnp.arange(3))
        
        # 7. Smooth driving bonus (penalize erratic behavior)
        omega_penalty = -jnp.abs(e_after) * jnp.abs(vx_after) * 0.01
        
        # Combine all rewards
        total_reward = (
            progress_reward + 
            speed_reward + 
            centerline_reward + 
            improvement_bonus +
            hard_zone_penalty + 
            outside_penalty + 
            racing_rewards +
            omega_penalty
        )
        
        # Store components for debugging
        info = {
            'progress': progress_reward,
            'speed': speed_reward,
            'centerline': centerline_reward,
            'improvement': improvement_bonus,
            'hard_zone': hard_zone_penalty,
            'outside': outside_penalty,
            'racing': racing_rewards,
            'smooth': omega_penalty
        }
        
        return total_reward, info
    
    def normalize_observations(feats, self_i):
        """Enhanced observation normalization."""
        a = (self_i + 1) % 3
        b = (self_i + 2) % 3
        
        # Find front car
        da = jnp.abs(wrap_diff(feats[a,0], feats[self_i,0], track_L))
        db = jnp.abs(wrap_diff(feats[b,0], feats[self_i,0], track_L))
        front_idx = jnp.where(da <= db, a, b)
        
        front = jnp.take(feats, front_idx, axis=0)
        fself = jnp.take(feats, self_i, axis=0)
        
        # Normalization functions
        def ang(x): return jnp.clip(x / jnp.pi, -1., 1.)
        def spd(x): return jnp.clip(x / vmax_mps, -1., 1.)
        def lat(x): return jnp.clip(x / lane_half_width_m, -1., 1.)
        
        base_obs = jnp.array([
            # Relative position (normalized)
            wrap_diff(front[0], fself[0], track_L) / track_L,
            
            # Lateral positions
            lat(front[1]), lat(fself[1]),
            
            # Heading differences
            ang(front[2]), ang(fself[2]),
            
            # Velocities
            spd(front[3]), spd(front[4]), 
            spd(fself[3]), spd(fself[4]),
            
            # Angular velocities (important for control)
            front[5] * 0.1, fself[5] * 0.1,
            
            # Curvatures (track information)
            front[6], fself[6],
            front[7], fself[7],
            
            # Additional useful features
            jnp.tanh(fself[1] / 0.5),  # Lateral position indicator
            jnp.sign(fself[1]),  # Which side of track
        ], dtype=jnp.float32)
        
        return base_obs
    
    def jax_reset(key: jax.Array) -> Tuple[EnvState, jnp.ndarray]:
        """Reset environment to initial state."""
        poses = spawn_poses()
        
        # Add slight randomization to starting positions for variety
        if training_mode:
            key, subkey = jax.random.split(key)
            pose_noise = jax.random.normal(subkey, (3, 3)) * 0.1
            pose_noise = pose_noise.at[:, 2].set(pose_noise[:, 2] * 0.05)  # Less heading noise
            poses = poses + pose_noise
        
        cars = CarBatchState(
            x=poses[:,0], y=poses[:,1], psi=poses[:,2],
            vx=jnp.zeros(3), vy=jnp.zeros(3), omega=jnp.zeros(3)
        )
        
        delay_buf = jnp.zeros((3, delay, 2), dtype=jnp.float32)
        
        # Compute initial features
        feats0 = compute_features(cars)
        
        # Compute initial relative positions
        def rel_for(i):
            a = (i + 1) % 3
            b = (i + 2) % 3
            s_self = feats0[i, 0]
            s_a = feats0[a, 0]
            s_b = feats0[b, 0]
            return wrap_diff(s_self, jnp.maximum(s_a, s_b), track_L)
        
        last_rel = jax.vmap(rel_for)(jnp.arange(3))
        
        state = EnvState(
            cars=cars,
            delay_buf=delay_buf,
            t=jnp.array(0, jnp.int32),
            last_rel=last_rel,
            track_L=jnp.asarray(track_L, jnp.float32)
        )
        
        # Build initial observations
        obs0 = jax.vmap(lambda i: normalize_observations(feats0, i))(jnp.arange(3))
        
        return state, obs0
    
    def jax_step(state: EnvState, action: jnp.ndarray):
        """
        Step environment forward.
        
        Args:
            state: Current environment state
            action: (3, 2) array of [velocity, steering] commands
        """
        action = jnp.clip(action, -1.0, 1.0)
        
        # Get current features and references
        feats_before = compute_features(state.cars)
        refs = get_references(state.cars)
        psi_ref = refs[:, 0]
        speed_ref = jnp.clip(refs[:, 1], 0.0, 1.0)
        
        # Compute reference control
        hdg_err = angle_diff(psi_ref, state.cars.psi)
        e = feats_before[:, 1]
        vx_b = jnp.maximum(feats_before[:, 3], 0.0)
        
        # Base reference steering
        steer_ref = 1.0 * hdg_err + jnp.arctan2(0.8 * (-e), vx_b + 0.2)
        steer_ref = jnp.clip(steer_ref, -1.0, 1.0)
        
        # Apply residual control with larger influence
        res_vel = action[:, 0] * 0.40  # Increased from 0.30
        res_steer = action[:, 1] * 0.50  # Increased from 0.40
        
        vel_cmd = jnp.clip(speed_ref + res_vel, 0.0, 1.0)
        steer_cmd = jnp.clip(steer_ref + res_steer, -1.0, 1.0)
        
        a_for_filter = jnp.stack([vel_cmd, steer_cmd], axis=1)
        
        # Apply improved safety filter
        prev_cmd = state.delay_buf[:, -1, :] if state.delay_buf.shape[1] > 0 else None
        a_safe, in_hard, outside = improved_safety_filter(
            a_for_filter, feats_before, lane_half_width_m,
            psi_ref, state.cars.psi, prev_cmd
        )
        
        # Handle delay buffer
        a0 = jnp.stack([
            jnp.clip(a_safe[:, 0], 0., 1.),
            jnp.clip(a_safe[:, 1], -1., 1.)
        ], axis=1)
        
        if state.delay_buf.shape[1] > 0:
            buf1 = jnp.concatenate([a0[:, None, :], state.delay_buf[:, :-1, :]], axis=1)
            cmd_prev = buf1[:, -1, :]
            # Reduce delay impact in hard zones
            cmd = jnp.where(in_hard[:, None], a0, cmd_prev)
        else:
            buf1 = state.delay_buf
            cmd = a0
        
        target_vel, target_steer = cmd[:, 0], cmd[:, 1]
        
        # Integrate dynamics
        S = state.cars
        nx, ny, npsi, nvx, nvy, nomega = rk4_step(
            params, (S.x, S.y, S.psi, S.vx, S.vy, S.omega),
            target_vel, target_steer
        )
        
        cars2 = CarBatchState(x=nx, y=ny, psi=npsi, vx=nvx, vy=nvy, omega=nomega)
        
        # Compute features after step
        feats_after = compute_features(cars2)
        
        # Build observations
        obs_before = jax.vmap(lambda i: normalize_observations(feats_before, i))(jnp.arange(3))
        next_obs = jax.vmap(lambda i: normalize_observations(feats_after, i))(jnp.arange(3))
        
        # Compute relative positions
        def rel_for(feats, i):
            aidx = (i + 1) % 3
            bidx = (i + 2) % 3
            s_self = feats[i, 0]
            s_a = feats[aidx, 0]
            s_b = feats[bidx, 0]
            return wrap_diff(s_self, jnp.maximum(s_a, s_b), track_L)
        
        rel_after = jax.vmap(lambda i: rel_for(feats_after, i))(jnp.arange(3))
        
        # Compute improved rewards
        s_before = feats_before[:, 0]
        s_after = feats_after[:, 0]
        e_before = feats_before[:, 1]
        e_after = feats_after[:, 1]
        vx_before = feats_before[:, 3]
        vx_after = feats_after[:, 3]
        
        rewards, reward_info = compute_improved_reward(
            s_before, s_after, e_before, e_after,
            vx_before, vx_after, in_hard, outside,
            state.last_rel, rel_after,
            track_L, lane_half_width_m
        )
        
        # Check termination
        t2 = state.t + jnp.int32(1)
        done = t2 >= jnp.int32(EP_LEN)
        truncated = done
        
        # Update state
        state2 = EnvState(
            cars=cars2,
            delay_buf=buf1,
            t=t2,
            last_rel=rel_after,
            track_L=track_L
        )
        
        # Return with info for compatibility (but simplified)
        info_obs_before = obs_before
        
        return state2, next_obs, rewards, done, truncated, info_obs_before
    
    return jax_reset, jax_step


def build_step_and_reset(num_envs, training_mode=True):
    """
    Build improved environment WITHOUT curriculum learning.
    
    Args:
        num_envs: Number of parallel environments
        training_mode: Whether in training mode (allows slightly relaxed safety)
    """
    # Fixed parameters - start with reasonable difficulty
    params = DynamicParams(
        num_envs=num_envs,
        DT=0.1,
        Sa=0.34,  # Fixed steering sensitivity
        Sb=0.0,
        Ta=20.,
        Tb=0.,
        mu=0.6,   # Fixed friction
        delay=2   # Fixed delay - reduced from 4 for better learning
    )
    
    # Load track
    path_rn = "/Users/sanikabharvirkar/Documents/alpha-RACER/simulators/params-num.yaml"
    path = load_path(path_rn)
    spec = init_waypoints(
        kind='custom',
        dt=0.1,
        H=9,
        speed=1.0,
        path=jnp.array(path),
        scale=6.5
    )
    track_L = float(path[-1, 0])
    
    def wp_generate(obs5, vx):
        targets, kin_pos, s, e = generate(
            spec, obs5, dt=0.1, mu_factor=1.0, body_speed=vx
        )
        if targets.shape[1] == 4:
            zeros = jnp.zeros((targets.shape[0], 1))
            targets = jnp.concatenate([targets[:, :3], zeros, targets[:, 3:4]], axis=1)
        return targets, kin_pos, s, e
    
    # Build improved environment
    reset_fn, step_fn = build_env_functions(
        params, 
        EP_LEN=500,
        track_L=float(track_L),
        delay=params.delay,
        wp_generate=wp_generate,
        training_mode=training_mode
    )
    
    # JIT compile
    reset_jit = jax.jit(reset_fn)
    step_jit = jax.jit(step_fn, donate_argnums=(0,))
    
    return reset_jit, step_jit


# Example usage for monitoring training
class TrainingMonitor:
    """Helper class to monitor training progress."""
    
    def __init__(self, log_freq=100):
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.safety_violations = []
        self.avg_speeds = []
        self.step_count = 0
    
    def log_episode(self, rewards, lengths, violations, speeds):
        """Log episode statistics."""
        self.episode_rewards.extend(rewards)
        self.episode_lengths.extend(lengths)
        self.safety_violations.extend(violations)
        self.avg_speeds.extend(speeds)
        self.step_count += sum(lengths)
        
        if self.step_count % self.log_freq == 0:
            self.print_stats()
    
    def print_stats(self):
        """Print recent statistics."""
        if len(self.episode_rewards) > 0:
            recent = min(100, len(self.episode_rewards))
            print(f"\n=== Step {self.step_count} ===")
            print(f"Avg Reward: {np.mean(self.episode_rewards[-recent:]):.2f}")
            print(f"Avg Length: {np.mean(self.episode_lengths[-recent:]):.2f}")
            print(f"Avg Speed: {np.mean(self.avg_speeds[-recent:]):.3f}")
            print(f"Safety Violations: {np.mean(self.safety_violations[-recent:]):.1%}")
