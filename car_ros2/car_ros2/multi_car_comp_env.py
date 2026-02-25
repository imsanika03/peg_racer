import scipy.interpolate
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32, TwistStamped
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
import random
import yaml

import numpy as np
import matplotlib.pyplot as plt

from tf_transformations import quaternion_from_euler, euler_matrix
import casadi as ca


from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.controllers_torch import rollout_fn_select as rollout_fn_select_torch
from car_dynamics.controllers_torch import reward_track_fn, MPPIController as MPPIControllerTorch
from car_dynamics.controllers_jax import MPPIController, MPPIParams, rollout_fn_select
from car_dynamics.envs.car3 import OffroadCar
from car_dynamics.controllers_jax import WaypointGenerator
from std_msgs.msg import Float64, Int8
import torch.nn as nn
import torch.optim as optim
import torch
import time
import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
import pickle
import os
from model_arch import SimpleModel
from ackermann_msgs.msg import AckermannDrive
import tf_transformations
import socket
import struct
import threading
import argparse
import scipy
from mpc_controller import mpc
from multi_car_comp_blocking import EP_LEN, CarNode, DT_torch, vehicle_dynamics, DT, DELAY, H, i_start
# from stable_baselines3 import PPO
print("DEVICE", jax.devices())

USE_OURS = True
RECORD_RACES = True
K_linear = 100
N_steps_per_iter = 6
grad_rate = 0.00001
# IBR params
N = 10
dt_ibr = 0.1
MU_FACTOR = 1.


## assuming numerical SIM

class MultiCarCompEnv(CarNode):

    agents = ["car0", "car1", "car2"]

    def __init__(self, trajectory_type="../../simulators/params-num.yaml"):

        super().__init__()
        self.timer_.cancel()
        self.slow_timer_.cancel()

        self.model_params_single = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)
        self.model_params_single_opp = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)
        self.model_params_single_opp1 = DynamicParams(num_envs=1, DT=DT,Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5,delay=DELAY)

        self.dynamics_single = DynamicBicycleModel(self.model_params_single)
        self.dynamics_single_opp = DynamicBicycleModel(self.model_params_single_opp)
        self.dynamics_single_opp1 = DynamicBicycleModel(self.model_params_single_opp1)

        self.waypoint_generator = WaypointGenerator(trajectory_type, DT, H, 2.)
        self.waypoint_generator_opp = WaypointGenerator(trajectory_type, DT, H, 1.)
        self.waypoint_generator_opp1 = WaypointGenerator(trajectory_type, DT, H, 1.)

        self.env = OffroadCar({}, self.dynamics_single)
        self.env_opp = OffroadCar({}, self.dynamics_single_opp)
        self.env_opp1 = OffroadCar({}, self.dynamics_single_opp1)

    def obs_state(self):
        return self.env.obs_state()

    def obs_state_opp(self):
        return self.env_opp.obs_state()

    def obs_state_opp1(self):
        return self.env_opp1.obs_state()

    def _wrap_s(self, v):
        if v < -75.: v += 150.
        if v > 75.: v -= 150.
        return v

    def reset(self):

        self.dynamics_single.reset()
        self.dynamics_single_opp.reset()
        self.dynamics_single_opp1.reset()

        self.obs = self.env.reset(pose=[3.,5.,-np.pi/2.-0.72])
        self.obs_opp = self.env_opp.reset(pose=[0.,0.,-np.pi/2.-0.5])
        self.obs_opp1 = self.env_opp1.reset(pose=[-2.,-6.,-np.pi/2.-0.5])

        self.waypoint_generator.last_i = -1
        self.waypoint_generator_opp.last_i = -1
        self.waypoint_generator_opp1.last_i = -1
        self.last_i = -1
        self.last_i_opp = -1
        self.last_i_opp1 = -1
        self.i = 0
        self.n_wins = [0,0,0]
        self.s = 0.
        self.s_opp = 0.
        self.s_opp1 = 0.

        return self._build_all_obs()

    def reset_params(self):

        self.last_i = -1
        self.last_i_opp = -1
        self.last_i_opp1 = -1
        self.ep_no += 1
        self.i = 1
        self.waypoint_generator.last_i = -1
        self.waypoint_generator_opp.last_i = -1
        self.waypoint_generator_opp1.last_i = -1
        if self.ep_no < 34 :
            obs = self.env.reset(pose=[3.,5.,-np.pi/2.-0.72])
            obs_opp1 = self.env_opp1.reset(pose=[0.,0.,-np.pi/2.-0.5])
            obs_opp = self.env_opp.reset(pose=[-2.,-6.,-np.pi/2.-0.5])
        elif self.ep_no < 67 :
            obs_opp1 = self.env_opp1.reset(pose=[3.,5.,-np.pi/2.-0.72])
            obs = self.env.reset(pose=[0.,0.,-np.pi/2.-0.5])
            obs_opp = self.env_opp.reset(pose=[-2.,-6.,-np.pi/2.-0.72])
        else :
            obs_opp1 = self.env_opp1.reset(pose=[3.,5.,-np.pi/2.-0.72])
            obs_opp = self.env_opp.reset(pose=[0.,0.,-np.pi/2.-0.5])
            obs = self.env.reset(pose=[-2.,-6.,-np.pi/2.-0.72])
        self.curr_sf1 = np.random.uniform(0.1,0.5)
        self.curr_sf2 = np.random.uniform(0.1,0.5)
        self.curr_lookahead_factor = np.random.uniform(0.12,0.5)
        self.curr_speed_factor = np.random.uniform(0.85,1.1)
        self.blocking = np.random.uniform(0.,1.0)

        self.curr_sf1_opp = np.random.uniform(0.1,0.5)
        self.curr_sf2_opp = np.random.uniform(0.1,0.5)
        self.curr_lookahead_factor_opp = np.random.uniform(0.12,0.5)
        self.curr_speed_factor_opp = np.random.uniform(0.85,1.1)
        self.blocking_opp = np.random.uniform(0.,1.0)

        self.curr_sf1_opp1 = np.random.uniform(0.1,0.5)
        self.curr_sf2_opp1 = np.random.uniform(0.1,0.5)
        self.curr_lookahead_factor_opp1 = np.random.uniform(0.12,0.5)
        self.curr_speed_factor_opp1 = np.random.uniform(0.85,1.1)
        self.blocking_opp1 = np.random.uniform(0.,1.0)

        self.curr_sf1_opp_ = np.random.uniform(0.1,0.5)
        self.curr_sf2_opp_ = np.random.uniform(0.1,0.5)
        self.curr_lookahead_factor_opp_ = np.random.uniform(0.12,0.5)
        self.curr_speed_factor_opp_ = np.random.uniform(0.85,1.1)
        self.blocking_opp_ = np.random.uniform(0.,1.0)

        self.curr_sf1_opp1_ = np.random.uniform(0.1,0.5)
        self.curr_sf2_opp1_ = np.random.uniform(0.1,0.5)
        self.curr_lookahead_factor_opp1_ = np.random.uniform(0.12,0.5)
        self.curr_speed_factor_opp1_ = np.random.uniform(0.85,1.1)
        self.blocking_opp1_ = np.random.uniform(0.,1.0)

    def _build_all_obs(self):
        px, py, psi, vx, vy, omega = self.obs_state().tolist()
        px_opp, py_opp, psi_opp, vx_opp, vy_opp, omega_opp = self.obs_state_opp().tolist()
        px_opp1, py_opp1, psi_opp1, vx_opp1, vy_opp1, omega_opp1 = self.obs_state_opp1().tolist()

        target_pos_tensor, _, s, e = self.waypoint_generator.generate(jnp.array(self.obs[:5]),dt=DT_torch,mu_factor=MU_FACTOR,body_speed=vx)
        target_pos_tensor_opp, _, s_opp, e_opp = self.waypoint_generator_opp.generate(jnp.array(self.obs_opp[:5]),dt=DT_torch,mu_factor=MU_FACTOR,body_speed=vx_opp)
        target_pos_tensor_opp1, _, s_opp1, e_opp1 = self.waypoint_generator_opp1.generate(jnp.array(self.obs_opp1[:5]),dt=DT_torch,mu_factor=MU_FACTOR,body_speed=vx_opp1)

        curv = float(target_pos_tensor[0,3])
        curv_opp = float(target_pos_tensor_opp[0,3])
        curv_opp1 = float(target_pos_tensor_opp1[0,3])
        curv_lookahead = float(target_pos_tensor[-1,3])
        curv_opp_lookahead = float(target_pos_tensor_opp[-1,3])
        curv_opp1_lookahead = float(target_pos_tensor_opp1[-1,3])

        target_pos_list = np.array(target_pos_tensor)
        theta_diff = float(np.arctan2(np.sin(target_pos_list[0,2]-psi),np.cos(target_pos_list[0,2]-psi)))
        theta_diff_opp = float(np.arctan2(np.sin(float(target_pos_tensor_opp[0,2])-psi_opp),np.cos(float(target_pos_tensor_opp[0,2])-psi_opp)))
        theta_diff_opp1 = float(np.arctan2(np.sin(float(target_pos_tensor_opp1[0,2])-psi_opp1),np.cos(float(target_pos_tensor_opp1[0,2])-psi_opp1)))

        self.s = float(s)
        self.s_opp = float(s_opp)
        self.s_opp1 = float(s_opp1)

        self._last_wp = {
            's': float(s), 'e': float(e), 's_opp': float(s_opp), 'e_opp': float(e_opp), 's_opp1': float(s_opp1), 'e_opp1': float(e_opp1),
            'theta_diff': theta_diff, 'theta_diff_opp': theta_diff_opp, 'theta_diff_opp1': theta_diff_opp1,
            'curv': curv, 'curv_opp': curv_opp, 'curv_opp1': curv_opp1,
            'curv_lookahead': curv_lookahead, 'curv_opp_lookahead': curv_opp_lookahead, 'curv_opp1_lookahead': curv_opp1_lookahead,
            'vx': vx, 'vy': vy, 'omega': omega,
            'vx_opp': vx_opp, 'vy_opp': vy_opp, 'omega_opp': omega_opp,
            'vx_opp1': vx_opp1, 'vy_opp1': vy_opp1, 'omega_opp1': omega_opp1,
            'px': px, 'py': py, 'psi': psi,
            'px_opp': px_opp, 'py_opp': py_opp, 'psi_opp': psi_opp,
            'px_opp1': px_opp1, 'py_opp1': py_opp1, 'psi_opp1': psi_opp1,
        }

        gap_01 = self._wrap_s(s_opp - s)
        gap_02 = self._wrap_s(s_opp1 - s)
        gap_10 = self._wrap_s(s - s_opp)
        gap_12 = self._wrap_s(s_opp1 - s_opp)
        gap_20 = self._wrap_s(s - s_opp1)
        gap_21 = self._wrap_s(s_opp - s_opp1)

        observations = {
            "car0": np.array([e, theta_diff, vx, vy, omega,
                              gap_01, e_opp, theta_diff_opp, vx_opp, vy_opp, omega_opp,
                              gap_02, e_opp1, theta_diff_opp1, vx_opp1, vy_opp1, omega_opp1,
                              curv, curv_lookahead, curv_opp, curv_opp1]),
            "car1": np.array([e_opp, theta_diff_opp, vx_opp, vy_opp, omega_opp,
                              gap_10, e, theta_diff, vx, vy, omega,
                              gap_12, e_opp1, theta_diff_opp1, vx_opp1, vy_opp1, omega_opp1,
                              curv_opp, curv_opp_lookahead, curv, curv_opp1]),
            "car2": np.array([e_opp1, theta_diff_opp1, vx_opp1, vy_opp1, omega_opp1,
                              gap_20, e, theta_diff, vx, vy, omega,
                              gap_21, e_opp, theta_diff_opp, vx_opp, vy_opp, omega_opp,
                              curv_opp1, curv_opp1_lookahead, curv, curv_opp]),
        }
        return observations

    def step(self, actions):
        self.i += 1

        w = self._last_wp
        s_before = self.s
        s_opp_before = self.s_opp
        s_opp1_before = self.s_opp1

        px, py, psi = w['px'], w['py'], w['psi']
        px_opp, py_opp, psi_opp = w['px_opp'], w['py_opp'], w['psi_opp']
        px_opp1, py_opp1, psi_opp1 = w['px_opp1'], w['py_opp1'], w['psi_opp1']
        s, e, vx = w['s'], w['e'], w['vx']
        s_opp, e_opp, vx_opp = w['s_opp'], w['e_opp'], w['vx_opp']
        s_opp1, e_opp1, vx_opp1 = w['s_opp1'], w['e_opp1'], w['vx_opp1']
        theta_diff = w['theta_diff']
        theta_diff_opp = w['theta_diff_opp']
        theta_diff_opp1 = w['theta_diff_opp1']

        # car0
        a0 = actions["car0"]
        steer, throttle, _, _, self.last_i = self.mpc((px,py,psi),
                                                      (s,e,vx),
                                                      (s_opp,e_opp,vx_opp),
                                                      (s_opp1,e_opp1,vx_opp1),
                                                      a0[0],a0[1],a0[2]*2,a0[3]**2,a0[4],
                                                      last_i=self.last_i)
        if abs(e) > 0.55 :
            self.env.state.vx *= np.exp(-3*(abs(e)-0.55))
            self.env.state.vy *= np.exp(-3*(abs(e)-0.55))
            self.env.state.psi += (1-np.exp(-(abs(e)-0.55)))*(theta_diff)
            steer += (-np.sign(e) - steer)*(1-np.exp(-3*(abs(e)-0.55)))
        if abs(theta_diff) > 1. :
            throttle+=0.2
        obs, _, _, _ = self.env.step(np.array([throttle,steer]))
        self.obs = obs

        # car1
        a1 = actions["car1"]
        steer, throttle, _, _, self.last_i_opp = self.mpc((px_opp,py_opp,psi_opp),(s_opp,e_opp,vx_opp),(s,e,vx),(s_opp1,e_opp1,vx_opp1),a1[0],a1[1],a1[2]*2,a1[3]**2,a1[4],last_i=self.last_i_opp)
        if abs(e_opp) > 0.55 :
            self.env_opp.state.vx *= np.exp(-3*(abs(e_opp)-0.55))
            self.env_opp.state.vy *= np.exp(-3*(abs(e_opp)-0.55))
            self.env_opp.state.psi += (1-np.exp(-(abs(e_opp)-0.55)))*(theta_diff_opp)
            steer += (-np.sign(e_opp) - steer)*(1-np.exp(-3*(abs(e_opp)-0.55)))
        if abs(theta_diff_opp) > 1. :
            throttle+=0.2
        obs_opp, _, _, _ = self.env_opp.step(np.array([throttle,steer]))
        self.obs_opp = obs_opp

        # car2
        a2 = actions["car2"]
        steer, throttle, _, _, self.last_i_opp1 = self.mpc((px_opp1,py_opp1,psi_opp1),(s_opp1,e_opp1,vx_opp1),(s,e,vx),(s_opp,e_opp,vx_opp),a2[0],a2[1],a2[2]*2,a2[3]**2,a2[4],last_i=self.last_i_opp1)
        if abs(e_opp1) > 0.55 :
            self.env_opp1.state.vx *= np.exp(-3*(abs(e_opp1)-0.55))
            self.env_opp1.state.vy *= np.exp(-3*(abs(e_opp1)-0.55))
            self.env_opp1.state.psi += (1-np.exp(-(abs(e_opp1)-0.55)))*(theta_diff_opp1)
            steer += (-np.sign(e_opp1) - steer)*(1-np.exp(-3*(abs(e_opp1)-0.55)))
        if abs(theta_diff_opp1) > 1. :
            throttle+=0.2
        obs_opp1, _, _, _ = self.env_opp1.step(np.array([throttle,steer]))
        self.obs_opp1 = obs_opp1

        # collisions
        collision = self.has_collided(px,py,psi,px_opp,py_opp,psi_opp)
        collision1 = self.has_collided(px,py,psi,px_opp1,py_opp1,psi_opp1)
        collision2 = self.has_collided(px_opp,py_opp,psi_opp,px_opp1,py_opp1,psi_opp1)

        diff_s = self._wrap_s(s_opp - s)
        if diff_s > 0. :
            self.env.state.vx *= np.exp(-20*collision)
            self.env.state.vy *= np.exp(-20*collision)
            self.env_opp.state.vx *= np.exp(-5*collision)
            self.env_opp.state.vy *= np.exp(-5*collision)
        else :
            self.env.state.vx *= np.exp(-5*collision)
            self.env.state.vy *= np.exp(-5*collision)
            self.env_opp.state.vx *= np.exp(-20*collision)
            self.env_opp.state.vy *= np.exp(-20*collision)

        diff_s1 = self._wrap_s(s_opp1 - s)
        if diff_s1 > 0. :
            self.env.state.vx *= np.exp(-20*collision1)
            self.env.state.vy *= np.exp(-20*collision1)
            self.env_opp1.state.vx *= np.exp(-5*collision1)
            self.env_opp1.state.vy *= np.exp(-5*collision1)
        else :
            self.env.state.vx *= np.exp(-5*collision1)
            self.env.state.vy *= np.exp(-5*collision1)
            self.env_opp1.state.vx *= np.exp(-20*collision1)
            self.env_opp1.state.vy *= np.exp(-20*collision1)

        diff_s2 = self._wrap_s(s_opp1 - s_opp)
        if diff_s2 > 0. :
            self.env_opp.state.vx *= np.exp(-20*collision2)
            self.env_opp.state.vy *= np.exp(-20*collision2)
            self.env_opp1.state.vx *= np.exp(-5*collision2)
            self.env_opp1.state.vy *= np.exp(-5*collision2)
        else :
            self.env_opp.state.vx *= np.exp(-5*collision2)
            self.env_opp.state.vy *= np.exp(-5*collision2)
            self.env_opp1.state.vx *= np.exp(-20*collision2)
            self.env_opp1.state.vy *= np.exp(-20*collision2)

        # observations (also updates self.s, self.s_opp, self.s_opp1)
        observations = self._build_all_obs()

        # rewards = delta s
        rewards = {
            "car0": self._wrap_s(self.s - s_before),
            "car1": self._wrap_s(self.s_opp - s_opp_before),
            "car2": self._wrap_s(self.s_opp1 - s_opp1_before),
        }

        done = self.i >= EP_LEN
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {"car0": {'s': self.s}, "car1": {'s': self.s_opp}, "car2": {'s': self.s_opp1}}

        return observations, rewards, terminations, truncations, infos
