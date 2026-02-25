import math, numpy as np
import gymnasium as gym
from gym.spaces import Box
import jax.numpy as jnp

from car_dynamics.envs.car3 import OffroadCar
from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.controllers_jax import WaypointGenerator

DT = 0.1
EP_LEN = 500


DT = 0.1
EP_LEN = 500

def _wrap_diff(a, b, L=75.08):
    d = a - b
    if d < -L/2: d += L
    if d >  L/2: d -= L
    return d

class RLMultiFromCarsEnv(gym.Env):


    def __init__(self, trajectory="berlin_2018", control="car1", opponents=None):

        super().__init__()

        self.control = control
        self.opponents = opponents or {}

        p = lambda: DynamicParams(num_envs=1, DT=DT, Sa=0.34, Sb=0., Ta=20., Tb=0., mu=0.5, delay=1)

        self.car0 = OffroadCar({}, DynamicBicycleModel(p()))
        self.car1 = OffroadCar({}, DynamicBicycleModel(p()))
        self.car2 = OffroadCar({}, DynamicBicycleModel(p()))

        self.wp0 = WaypointGenerator(trajectory, DT, 9, 1.0)
        self.wp1 = WaypointGenerator(trajectory, DT, 9, 1.0)
        self.wp2 = WaypointGenerator(trajectory, DT, 9, 1.0)

        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self.oracle_observation_space = Box(low=-np.inf, high=np.inf, shape=(51,), dtype=np.float32)

        self.t = 0
        self._last_s = None
        self._last_rel = None


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

        return dict(s=float(s), 
                    e=float(e),
                    theta_diff=float(math.atan2(math.sin(theta-psi), math.cos(theta-psi))),
                    curv=float(tgt[0,3]), 
                    curv_lh=float(tgt[-1,3]),
                    px=px, 
                    py=py, 
                    vx=vx, 
                    vy=vy, 
                    omega=omega)


    def _scripted(self, f):
        steer = np.clip(-f["theta_diff"], -1, 1)
        v_tgt = max(0.0, 4.0 - 6.0*abs(f["curv_lh"]))
        throttle = np.clip(v_tgt - f["vx"], -1, 1)
        return np.array([throttle, steer], dtype=np.float32)

    def _rl_obs(self, f_self, f_a, f_b):

        if abs(_wrap_diff(f_self["s"], f_a["s"], self.track_L)) < abs(_wrap_diff(f_a["s"], f_b["s"], self.track_L)):

            return np.array([f_a["s"]-f_self["s"], 
                             f_a["e"], f_self["e"],
                             f_a["theta_diff"], 
                             f_a["vx"], 
                             f_a["vy"], 
                             f_a["omega"],
                             f_self["theta_diff"], 
                             f_self["vx"], 
                             f_self["vy"], 
                             f_self["omega"],
                             f_a["curv"], 
                             f_self["curv"], 
                             f_a["curv_lh"], 
                             f_self["curv_lh"]], dtype=np.float32)
        else:

            return np.array([f_a["s"]-f_b["s"], 
                             f_a["e"], 
                             f_b["e"],
                             f_a["theta_diff"], 
                             f_a["vx"], 
                             f_a["vy"], 
                             f_a["omega"],
                             f_b["theta_diff"], 
                             f_b["vx"], 
                             f_b["vy"], 
                             f_b["omega"],
                             f_a["curv"], 
                             f_b["curv"], 
                             f_a["curv_lh"], 
                             f_b["curv_lh"]], dtype=np.float32)

    def _order(self):
  
        if self.control == "car0":
            return ("car0", self.o0, self.wp0), ("car1", self.o1, self.wp1), ("car2", self.o2, self.wp2)
        if self.control == "car1":
            return ("car1", self.o1, self.wp1), ("car0", self.o0, self.wp0), ("car2", self.o2, self.wp2)
        return ("car2", self.o2, self.wp2), ("car0", self.o0, self.wp0), ("car1", self.o1, self.wp1)
    
    def set_opponents(self, opponents_dict):
        self.opponents = opponents_dict or {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.spawn()

        # get ALL three ids from _order()
        (id_c, oc, wpc), (id_a, oa, wpa), (id_b, ob, wpb) = self._order()

        # compute features
        fc = self._feats(oc, wpc)
        fa = self._feats(oa, wpa)
        fb = self._feats(ob, wpb)
        feats = {id_c: fc, id_a: fa, id_b: fb}

        # legacy single-agent trackers (kept for your existing reward path)
        self._last_s = fc["s"]
        self._last_rel = _wrap_diff(fc["s"], max(fa["s"], fb["s"]), self.track_L)

        # init per-car last rels (used when you collect all cars’ rewards)
        if not hasattr(self, "_init_last_rel_all"):
            # fallback if helper isn't defined
            def _rel_for(feats, cid):
                others = [x for x in ("car0","car1","car2") if x != cid]
                f_self, f_a, f_b = feats[cid], feats[others[0]], feats[others[1]]
                return _wrap_diff(f_self["s"], max(f_a["s"], f_b["s"]), self.track_L)
            self._rel_for = _rel_for
            self._init_last_rel_all = lambda f: setattr(
                self, "_last_rel_by_id",
                {cid: self._rel_for(f, cid) for cid in ("car0","car1","car2")}
            )
        self._init_last_rel_all(feats)

        # controlled car's observation
        obs_c = self._obs_for(feats, id_c)

        # (optional) expose initial per-car obs in info
        info = {"all": {"obs": {
            "car0": self._obs_for(feats, "car0"),
            "car1": self._obs_for(feats, "car1"),
            "car2": self._obs_for(feats, "car2"),
        }}}
        return obs_c, info



    def step(self, action, collect_all=False):
        self.t += 1

        (id_c, oc, wpc), (id_a, oa, wpa), (id_b, ob, wpb) = self._order()

        # feats BEFORE stepping (for obs)
        fc = self._feats(oc, wpc)
        fa = self._feats(oa, wpa)
        fb = self._feats(ob, wpb)
        feats_before = {id_c: fc, id_a: fa, id_b: fb}

        # compute RL obs from each car's perspective BEFORE
        if collect_all:
            obs_all = {
                "car0": self._obs_for(feats_before, "car0"),
                "car1": self._obs_for(feats_before, "car1"),
                "car2": self._obs_for(feats_before, "car2"),
            }

        # actions for a/b (opponents)
        if id_a in self.opponents:
            act_a = self.opponents[id_a](self._obs_for(feats_before, id_a))
        else:
            act_a = self._scripted(fa)

        if id_b in self.opponents:
            act_b = self.opponents[id_b](self._obs_for(feats_before, id_b))
        else:
            act_b = self._scripted(fb)

        # action for controlled car
        act_c = np.asarray(action, dtype=np.float32).clip(-1, 1)

        # step each car
        oa, *_ = getattr(self, id_a).step(np.asarray(act_a, dtype=np.float32))
        ob, *_ = getattr(self, id_b).step(np.asarray(act_b, dtype=np.float32))
        oc, *_ = getattr(self, id_c).step(act_c)

        # feats AFTER stepping (for next_obs and reward updates)
        fc_n = self._feats(oc, wpc)
        fa_n = self._feats(oa, wpa)
        fb_n = self._feats(ob, wpb)
        feats_after = {id_c: fc_n, id_a: fa_n, id_b: fb_n}

        # controlled agent's obs after (your original 'obs' variable)
        obs_c = self._obs_for(feats_after, id_c)

        # controlled reward (same as before)
        rel = _wrap_diff(fc_n["s"], max(fa_n["s"], fb_n["s"]), self.track_L)
        reward = _wrap_diff(rel, self._last_rel, self.track_L)
        self._last_rel = rel

        terminated = False
        truncated = (self.t >= EP_LEN)

        info = {"rel_prog": reward, "rel": rel}

        if collect_all:
            # per-car rewards using per-car last_rel tracking
            reward_all = {}
            for cid in ("car0","car1","car2"):
                rel_c = self._rel_for(feats_after, cid)
                reward_all[cid] = _wrap_diff(rel_c, self._last_rel_by_id[cid], self.track_L)
                self._last_rel_by_id[cid] = rel_c

            info["all"] = {
                "obs": {
                    "car0": obs_all["car0"],
                    "car1": obs_all["car1"],
                    "car2": obs_all["car2"],
                },
                "action": {
                    "car0": act_c if id_c == "car0" else (act_a if id_a == "car0" else act_b),
                    "car1": act_c if id_c == "car1" else (act_a if id_a == "car1" else act_b),
                    "car2": act_c if id_c == "car2" else (act_a if id_a == "car2" else act_b),
                },
                "next_obs": {
                    "car0": self._obs_for(feats_after, "car0"),
                    "car1": self._obs_for(feats_after, "car1"),
                    "car2": self._obs_for(feats_after, "car2"),
                },
                "reward": reward_all,  # optional but handy
            }

        return obs_c, reward, terminated, truncated, info

    

    def _obs_for(self, feats, self_id):

        others = [cid for cid in ("car0","car1","car2") if cid != self_id]
        return self._rl_obs(feats[self_id], feats[others[0]], feats[others[1]])

    def _rel_for(self, feats, self_id):
        """Compute the 'rel' term (used for reward) for a given self_id."""
        others = [cid for cid in ("car0","car1","car2") if cid != self_id]
        f_self, f_a, f_b = feats[self_id], feats[others[0]], feats[others[1]]
        return _wrap_diff(f_self["s"], max(f_a["s"], f_b["s"]), self.track_L)

    # keep separate last_rel per car so we can compute per-car rewards if you want them
    def _init_last_rel_all(self, feats):
        self._last_rel_by_id = {cid: self._rel_for(feats, cid) for cid in ("car0","car1","car2")}


class SB3PolicyFn:

    def __init__(self, model): self.model = model
    def __call__(self, obs):
        a, _ = self.model.predict(obs, deterministic=False)
        return a