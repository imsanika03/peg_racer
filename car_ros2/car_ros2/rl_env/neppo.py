import math, numpy as np
# import gymnasium as gym
from gymnasium import spaces
import jax.numpy as jnp

from car_dynamics.envs.car3 import OffroadCar
from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.controllers_jax import WaypointGenerator

DT = 0.1
EP_LEN = 500

def _wrap_diff(a, b, L=75.08):
    d = a - b
    if d < -L/2: d += L
    if d >  L/2: d -= L
    return d

class AutoThreeCarEnv:

    def __init__(self, trajectory="berlin_2018"):

        p = lambda: DynamicParams(num_envs=1, DT=DT, Sa=0.34, Sb=0., Ta=20., Tb=0., mu=0.5, delay=1)

        self.car0 = OffroadCar({}, DynamicBicycleModel(p()))
        self.car1 = OffroadCar({}, DynamicBicycleModel(p()))
        self.car2 = OffroadCar({}, DynamicBicycleModel(p()))

        self.wp0 = WaypointGenerator(trajectory, DT, 9, 1.0)
        self.wp1 = WaypointGenerator(trajectory, DT, 9, 1.0)
        self.wp2 = WaypointGenerator(trajectory, DT, 9, 1.0)
        

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32) 

        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.oracle_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(51,), dtype=np.float32)

        self.t = 0

        if self.wp0.waypoint_type == 'custom':
            self.track_L = float(self.wp0.path[-1, 0])
        else:
            self.track_L = float(2.0 * np.pi)

    def spawn(self):
        self.o0 = self.car0.reset(pose=[ 3.,  5., -np.pi/2 - 0.72])
        self.o1 = self.car1.reset(pose=[ 0.,  0., -np.pi/2 - 0.50])
        self.o2 = self.car2.reset(pose=[-2., -6., -np.pi/2 - 0.50])

    def _feats(self, obs, wp):
        px, py, psi, vx, vy, omega = [float(x) for x in obs.tolist()]
        tgt, _, s, e = wp.generate(jnp.array(obs[:5]), dt=DT, mu_factor=1.0, body_speed=vx)
        theta = float(tgt[0,2])
        return dict(
            s=float(s), e=float(e),
            theta_diff=float(math.atan2(math.sin(theta-psi), math.cos(theta-psi))),
            curv=float(tgt[0,3]), curv_lh=float(tgt[-1,3]),
            px=px, py=py, vx=vx, vy=vy, omega=omega,
        )

    # obs relative to f_self (ego)
    def _rl_obs(self, f_self, f_a, f_b):

        da = abs(_wrap_diff(f_a["s"], f_self["s"], self.track_L))
        db = abs(_wrap_diff(f_b["s"], f_self["s"], self.track_L))
        front = f_a if da <= db else f_b

        return np.array([
            front["s"] - f_self["s"],
            front["e"],  f_self["e"],
            front["theta_diff"],
            front["vx"], front["vy"], front["omega"],
            f_self["theta_diff"],
            f_self["vx"], f_self["vy"], f_self["omega"],
            front["curv"],  f_self["curv"],
            front["curv_lh"], f_self["curv_lh"],
        ], dtype=np.float32)


    def _obs_for(self, feats, self_id):
        others = [cid for cid in ("car0","car1","car2") if cid != self_id]
        return self._rl_obs(feats[self_id], feats[others[0]], feats[others[1]])

    def _rel_for(self, feats, self_id):
        others = [cid for cid in ("car0","car1","car2") if cid != self_id]
        f_self, f_a, f_b = feats[self_id], feats[others[0]], feats[others[1]]
        return _wrap_diff(f_self["s"], max(f_a["s"], f_b["s"]), self.track_L)

    def _init_last_rel_all(self, feats):
        self._last_rel_by_id = {cid: self._rel_for(feats, cid) for cid in ("car0","car1","car2")}

    def reset(self, *, seed=None, options=None):

        self.t = 0
        self.spawn()

        f0 = self._feats(self.o0, self.wp0)
        f1 = self._feats(self.o1, self.wp1)
        f2 = self._feats(self.o2, self.wp2)
        feats = {"car0": f0, "car1": f1, "car2": f2}
        self._init_last_rel_all(feats)

        obs = [
            self._obs_for(feats, "car0"),
            self._obs_for(feats, "car1"),
            self._obs_for(feats, "car2"),
        ]
        return obs, {}

    def step(self, action):

        self.t += 1

        f0 = self._feats(self.o0, self.wp0)
        f1 = self._feats(self.o1, self.wp1)
        f2 = self._feats(self.o2, self.wp2)

        feats_before = {"car0": f0, "car1": f1, "car2": f2}
        obs_before = [self._obs_for(feats_before, "car0"), self._obs_for(feats_before, "car1"), self._obs_for(feats_before, "car2")]

        arr = np.asarray(action, dtype=np.float32)
        if arr.shape != (3, 2):
            raise ValueError(f"action must have shape (3,2), got {arr.shape}")
        a0, a1, a2 = arr[0].clip(-1,1), arr[1].clip(-1,1), arr[2].clip(-1,1)

        obs_car0, *_ = self.car0.step(a0)
        obs_car1, *_ = self.car1.step(a1)
        obs_car2, *_ = self.car2.step(a2)

        self.o0, self.o1, self.o2 = obs_car0, obs_car1, obs_car2


        f0n = self._feats(self.o0, self.wp0)
        f1n = self._feats(self.o1, self.wp1)
        f2n = self._feats(self.o2, self.wp2)

        feats_after = {"car0": f0n, "car1": f1n, "car2": f2n}

        next_obs = [
            self._obs_for(feats_after, "car0"),
            self._obs_for(feats_after, "car1"),
            self._obs_for(feats_after, "car2"),
        ]

        rewards = []
        for cid in ("car0","car1","car2"):
            rel_c = self._rel_for(feats_after, cid)
            r_c = _wrap_diff(rel_c, self._last_rel_by_id[cid], self.track_L)
            rewards.append(r_c)
            self._last_rel_by_id[cid] = rel_c

        terminated = False
        truncated = (self.t >= EP_LEN)

        info = {
            "obs": obs_before
        }

        return next_obs, rewards, terminated, truncated, info


# class AutoThreeCarVecEnv:
#     """Synchronous vector of AutoThreeCarEnv."""
#     def __init__(self, make_env_fn, num_envs: int):
#         self.envs = [make_env_fn() for _ in range(num_envs)]
#         self.num_envs = num_envs

#     def reset(self):
#         # obs: list length N, each is [car0_obs, car1_obs, car2_obs]
#         obs_infos = [env.reset() for env in self.envs]
#         obs, infos = zip(*obs_infos)  # obs: tuple of lists
#         obs = np.asarray([np.stack(o, axis=0) for o in obs], dtype=np.float32)  # (N, 3, obs_dim)
#         return obs, infos

#     def step(self, actions):
#         """actions: (N, 3, 2) -> next_obs: (N, 3, obs_dim), rewards: (N, 3)"""
#         N = self.num_envs
#         actions = np.asarray(actions, dtype=np.float32)
#         assert actions.shape[0] == N and actions.shape[1:] == (3, 2), f"actions shape must be (N,3,2), got {actions.shape}"

#         results = [self.envs[i].step(actions[i]) for i in range(N)]
#         next_obs, rewards, terms, truncs, infos = zip(*results)

#         next_obs = np.asarray([np.stack(o, axis=0) for o in next_obs], dtype=np.float32)  # (N, 3, obs_dim)
#         rewards  = np.asarray(rewards, dtype=np.float32)  # (N, 3)
#         terms    = np.asarray(terms, dtype=bool)          # (N,)
#         truncs   = np.asarray(truncs, dtype=bool)         # (N,)

#         # NOTE: caller handles per-env resets using terms|truncs
#         return next_obs, rewards, terms, truncs, infos

#     def reset_at(self, i):
#         """Reset a single env i that terminated/truncated mid-rollout."""
#         obs, info = self.envs[i].reset()
#         obs = np.stack(obs, axis=0).astype(np.float32)  # (3, obs_dim)
#         return obs, info

