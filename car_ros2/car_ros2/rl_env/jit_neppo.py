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
    delay: int = 4

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
    K1 = dbm_dxdt(*state, 
                  target_vel, 
                  target_steer,
                  params.Ta, 
                  params.Tb, 
                  params.Sa, 
                  params.Sb, 
                  params.LF, 
                  params.LR,
                  params.MASS, 
                  params.K_RFY,
                  params.K_FFY, 
                  params.Iz, 
                  params.mu,
                  params.Cf, 
                  params.Cr, 
                  params.Bf, 
                  params.Br, 
                  params.hcom, 
                  params.fr)
    
    S2 = (state[0] + 0.5*DT*K1[0],
          state[1] + 0.5*DT*K1[1],
          state[2] + 0.5*DT*K1[2],
          state[3] + 0.5*DT*K1[3],
          state[4] + 0.5*DT*K1[4],
          state[5] + 0.5*DT*K1[5])
    
    K2 = dbm_dxdt(*S2, 
                  target_vel, 
                  target_steer,
                  params.Ta,
                  params.Tb, 
                  params.Sa, 
                  params.Sb, 
                  params.LF, 
                  params.LR,
                  params.MASS, 
                  params.K_RFY, 
                  params.K_FFY, 
                  params.Iz, 
                  params.mu,
                  params.Cf, 
                  params.Cr, 
                  params.Bf, 
                  params.Br, 
                  params.hcom, 
                  params.fr)
    
    S3 = (state[0] + 0.5*DT*K2[0],
          state[1] + 0.5*DT*K2[1],
          state[2] + 0.5*DT*K2[2],
          state[3] + 0.5*DT*K2[3],
          state[4] + 0.5*DT*K2[4],
          state[5] + 0.5*DT*K2[5])
    
    K3 = dbm_dxdt(*S3, 
                  target_vel, 
                  target_steer,
                  params.Ta, 
                  params.Tb, 
                  params.Sa, 
                  params.Sb, 
                  params.LF, 
                  params.LR,
                  params.MASS, 
                  params.K_RFY, 
                  params.K_FFY, 
                  params.Iz, 
                  params.mu,
                  params.Cf, 
                  params.Cr, 
                  params.Bf, 
                  params.Br, 
                  params.hcom, 
                  params.fr)
    
    S4 = (state[0] + DT*K3[0],
          state[1] + DT*K3[1],
          state[2] + DT*K3[2],
          state[3] + DT*K3[3],
          state[4] + DT*K3[4],
          state[5] + DT*K3[5])
    
    K4 = dbm_dxdt(*S4, 
                  target_vel, 
                  target_steer,
                  params.Ta, 
                  params.Tb, 
                  params.Sa, 
                  params.Sb, 
                  params.LF, 
                  params.LR,
                  params.MASS, 
                  params.K_RFY, 
                  params.K_FFY, 
                  params.Iz, 
                  params.mu,
                  params.Cf, 
                  params.Cr, 
                  params.Bf, 
                  params.Br, 
                  params.hcom, 
                  params.fr)

    nx = state[0] + DT/6.0 * (K1[0] + 2*K2[0] + 2*K3[0] + K4[0])
    ny = state[1] + DT/6.0 * (K1[1] + 2*K2[1] + 2*K3[1] + K4[1])
    npsi = state[2] + DT/6.0 * (K1[2] + 2*K2[2] + 2*K3[2] + K4[2])
    nvx = state[3] + DT/6.0 * (K1[3] + 2*K2[3] + 2*K3[3] + K4[3])
    nvy = state[4] + DT/6.0 * (K1[4] + 2*K2[4] + 2*K3[4] + K4[4])
    nomega = state[5] + DT/6.0 * (K1[5] + 2*K2[5] + 2*K3[5] + K4[5])

    return (nx, ny, npsi, nvx, nvy, nomega)


def wrap_diff(a, b, L):
    d = a - b
    d = jnp.where(d < -L/2., d + L, d)
    d = jnp.where(d >  L/2., d - L, d)
    return d


def build_env_functions(params: DynamicParams,
                        EP_LEN: int,
                        track_L: float,
                        delay: int,
                        wp_generate):


    def spawn_poses():

        return jnp.array([
            [ 3.0,  5.0, -jnp.pi/2 - 0.72],
            [ 0.0,  0.0, -jnp.pi/2 - 0.50],
            [-2.0, -6.0, -jnp.pi/2 - 0.50],
        ], dtype=jnp.float32)

    def jax_reset(key: jax.Array) -> Tuple[EnvState, jnp.ndarray]:
        poses = spawn_poses()
        cars = CarBatchState(
            x=poses[:,0], y=poses[:,1], psi=poses[:,2],
            vx=jnp.zeros(3), vy=jnp.zeros(3), omega=jnp.zeros(3),
        )
        delay_buf = jnp.zeros((3, delay, 2), dtype=jnp.float32)


        def car_features(i, _):
            obs5 = jnp.array([cars.x[i], cars.y[i], cars.psi[i], cars.vx[i], cars.vy[i]])
            tgt, _, s, e = wp_generate(obs5, cars.vx[i])
            theta = tgt[0,2]
            theta_diff = jnp.arctan2(jnp.sin(theta - cars.psi[i]), jnp.cos(theta - cars.psi[i]))
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


        def rl_obs(self_i):
            a = (self_i + 1) % 3
            b = (self_i + 2) % 3
            da = jnp.abs(wrap_diff(feats0[a,0], feats0[self_i,0], track_L))
            db = jnp.abs(wrap_diff(feats0[b,0], feats0[self_i,0], track_L))
            front_idx = jnp.where(da <= db, a, b)

            front = jnp.take(feats0, front_idx, axis=0) 
            fself = jnp.take(feats0, self_i, axis=0)

            return jnp.array([
                front[0] - fself[0],
                front[1],  fself[1], 
                front[2],
                front[3], front[4], front[5], 
                fself[2],  
                fself[3], fself[4], fself[5],  
                front[6],  fself[6],  
                front[7],  fself[7], 
            ], dtype=jnp.float32)


        obs0 = jax.vmap(rl_obs)(jnp.arange(3))
        return state, obs0

    def jax_step(state: EnvState, action: jnp.ndarray):

        a = jnp.clip(action, -1.0, 1.0)

        a0 = jnp.stack([jnp.clip(a[:,0], 0., 1.), jnp.clip(a[:,1], -1., 1.)], axis=1)


        buf1 = jnp.concatenate([a0[:,None,:], state.delay_buf[:,:-1,:]], axis=1)
        cmd  = buf1[:,-1,:] 
        target_vel   = cmd[:,0]
        target_steer = cmd[:,1]

        # batched
        S = state.cars
        next_tuple = rk4_step(params,
            (S.x, S.y, S.psi, S.vx, S.vy, S.omega),
            target_vel, target_steer)

        cars2 = CarBatchState(
            x=next_tuple[0], y=next_tuple[1], psi=next_tuple[2],
            vx=next_tuple[3], vy=next_tuple[4], omega=next_tuple[5]
        )


        def feats_from(cars):
            def one(i):
                obs5 = jnp.array([cars.x[i], cars.y[i], cars.psi[i], cars.vx[i], cars.vy[i]])
                tgt, _, s, e = wp_generate(obs5, cars.vx[i])
                theta = tgt[0,2]
                theta_diff = jnp.arctan2(jnp.sin(theta - cars.psi[i]), jnp.cos(theta - cars.psi[i]))
                curv = tgt[0,3]
                curv_lh = tgt[-1,3]
                return jnp.array([s, e, theta_diff, cars.vx[i], cars.vy[i], cars.omega[i], curv, curv_lh], jnp.float32)
            
            return jax.vmap(one)(jnp.arange(3))  
        
        feats_before = feats_from(state.cars)
        feats_after  = feats_from(cars2)

        def obs_from_features(feats, self_i):
            a = (self_i + 1) % 3
            b = (self_i + 2) % 3
            da = jnp.abs(wrap_diff(feats[a,0], feats[self_i,0], track_L))
            db = jnp.abs(wrap_diff(feats[b,0], feats[self_i,0], track_L))
            front_idx = jnp.where(da <= db, a, b)

            front = jnp.take(feats, front_idx, axis=0) 
            fself = jnp.take(feats, self_i,   axis=0)

            return jnp.array([
                front[0] - fself[0],
                front[1],  fself[1],
                front[2],
                front[3], front[4], front[5],
                fself[2],
                fself[3], fself[4], fself[5],
                front[6],  fself[6],
                front[7],  fself[7],
            ], dtype=jnp.float32)


        obs_before = jax.vmap(lambda i: obs_from_features(feats_before, i))(jnp.arange(3))
        next_obs   = jax.vmap(lambda i: obs_from_features(feats_after,  i))(jnp.arange(3))

        def rel_for(feats, i):
            a = (i + 1) % 3
            b = (i + 2) % 3
            s_self = feats[i,0]
            s_a = feats[a,0]
            s_b = feats[b,0]
            return wrap_diff(s_self, jnp.maximum(s_a, s_b), track_L)


        rel_after = jax.vmap(lambda i: rel_for(feats_after, i))(jnp.arange(3))
        r = wrap_diff(rel_after, state.last_rel, track_L)

        t2 = state.t + jnp.int32(1)
        done = t2 >= jnp.int32(EP_LEN)
        truncated = done 

        state2 = EnvState(cars=cars2, 
                          delay_buf=buf1,
                          t=t2,
                          last_rel=rel_after, 
                          track_L=track_L)

        info_obs_before = obs_before
        return state2, next_obs, r, done, truncated, info_obs_before

    return jax_reset, jax_step

def load_path(waypoint_type):
        import os
        yaml_content = yaml.load(open(waypoint_type, 'r'), Loader=yaml.FullLoader)
        centerline_file = yaml_content['track_info']['centerline_file'][:-4]
        ox = yaml_content['track_info']['ox']
        oy = yaml_content['track_info']['oy']
        # Get the repository root directory (3 levels up from this file)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        ref_trajs_path = os.path.join(repo_root, 'ref_trajs', centerline_file + '_with_speeds.csv')
        df = pd.read_csv(ref_trajs_path)
        if waypoint_type.find('num') != -1:
            return np.array(df.iloc[:-1,:])*yaml_content['track_info']['scale'] + np.array([0, ox, oy, 0])
        else :
            return np.array(df.iloc[:,:]) + np.array([0, ox, oy, 0])



EP_LEN = 500

def build_step_and_reset(num_envs):
    import os
    params = DynamicParams(num_envs=num_envs, DT=0.1, Sa=0.34, Sb=0.0, Ta=20., Tb=0., mu=0.5, delay=4)

    # Get the repository root directory (3 levels up from this file)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path_rn = os.path.join(repo_root, 'simulators', 'params-num.yaml')
    path = load_path(path_rn)
    spec = init_waypoints(kind='custom', dt=0.1, H=9, speed=1.0, path=jnp.array(path), scale=6.5)
    track_L = float(path[-1,0])

    def wp_generate(obs5, vx):

        targets, kin_pos, s, e = generate(spec, obs5, dt=0.1, mu_factor=1.0, body_speed=vx)

        if targets.shape[1] == 4:
            zeros = jnp.zeros((targets.shape[0],1))
            targets = jnp.concatenate([targets[:,:3], zeros, targets[:,3:4]], axis=1)
        return targets, kin_pos, s, e


    reset_fn, step_fn = build_env_functions(params, EP_LEN, float(track_L), params.delay, wp_generate)

    reset_jit = jax.jit(reset_fn)
    step_jit = jax.jit(step_fn, donate_argnums=(0,))
    return reset_jit, step_jit
