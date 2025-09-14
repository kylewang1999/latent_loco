import sys, os, os.path as osp
os.environ["MUJOCO_GL"] = "egl"          # headless GPU
os.environ["PYOPENGL_PLATFORM"] = "egl"  # sometimes needed
import mujoco
from mujoco import mjx
from mujoco.mjx._src.types import Model, Data

import jax, jax.numpy as jnp
from jax import vmap, lax
import flax.nnx as nnx

from flax.struct import dataclass as flax_dataclass
from typing import Callable, Tuple, Optional, Dict


import dataclasses
from dataclasses import dataclass
from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union

from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from utils import load_xml_as_mjmodel, get_repo_root

@dataclass
class CfgEnvDoubleIntegrator:
    xml_path: str = os.path.join(get_repo_root(), 'assets/double_integrator.xml')
    mj_model_kwargs: dict = dataclasses.field(default_factory=lambda: {
        'solver': 'mjSOL_CG',
        'solver_iter': 6,
        'solver_ls_iter': 6,
    })
    pipeline_env_kwargs: dict = dataclasses.field(default_factory=lambda: {
        'n_frames': 5,
        'backend': 'mjx',
    })
    mj_solver: str = 'mjSOL_CG'
    mj_solver_iter: int = 6
    mj_solver_ls_iter: int = 6
    forward_reward_weight: float = 1.25
    ctrl_cost_weight: float = 0.1
    healthy_reward: float = 5.0
    terminate_when_unhealthy: bool = True
    healthy_x_range: Tuple[float, float] = (-20., 20.)
    healthy_y_range: Tuple[float, float] = (-20., 20.)
    reset_noise_scale: float = 5.


class DoubleIntegrator(PipelineEnv):

    def __init__(self,
                 cfg: CfgEnvDoubleIntegrator=CfgEnvDoubleIntegrator(),
                 **kwargs):
      
        self.cfg = cfg
        mj_model, sys = load_xml_as_mjmodel(cfg.xml_path, 
                                            mj_moodel_kwargs=cfg.mj_model_kwargs, 
                                            return_brax_sys=True)
        super().__init__(sys, **cfg.pipeline_env_kwargs)

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        ''' Add noise to  '''
        lo, hi = -self.cfg.reset_noise_scale, self.cfg.reset_noise_scale
        pos_noise = jax.random.uniform(rng1, (self.sys.nq,), minval=lo, maxval=hi)
        vel_noise = jax.random.uniform(rng2, (self.sys.nv,), minval=lo, maxval=hi)
        qpos = self.sys.qpos0 + pos_noise
        qvel = vel_noise

        ''' Cook up return State '''
        data = self.pipeline_init(qpos, qvel)  # only [qpos, qvel] matters for reset, others are zero placeholders
        obs = self.get_observation(data, jnp.zeros(self.sys.nu))
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        return State(data, obs, reward, done, metrics)


    def step(self, state: State, action: jnp.ndarray) -> State:
        data_curr = state.pipeline_state
        data_next = self.pipeline_step(data_curr, action)

        com_before = data_curr.subtree_com[1]
        com_after = data_next.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = jnp.clip(1/jnp.linalg.norm(com_after), max=5000)

        ''' Check if the agent is healthy '''
        x_min, x_max = self.cfg.healthy_x_range
        y_min, y_max = self.cfg.healthy_y_range
        is_healthy = (
            (data_next.q[0] >= x_min) & (data_next.q[0] <= x_max) &
            (data_next.q[1] >= y_min) & (data_next.q[1] <= y_max)
        ).astype(float) 
        if self.cfg.terminate_when_unhealthy:
            healthy_reward = self.cfg.healthy_reward
        else:
            healthy_reward = self.cfg.healthy_reward * is_healthy

        ctrl_cost = self.cfg.ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self.get_observation(data_next, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self.cfg.terminate_when_unhealthy else 0.0
        state.metrics.update(
            forward_reward=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jnp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )

        return state.replace(pipeline_state=data_next, obs=obs, reward=reward, done=done)

    def get_observation(self, data: mjx.Data, action: jnp.ndarray) -> jnp.ndarray:
        '''
        Returns:
            jnp.ndarray: []
        '''

        # external_contact_forces are excluded
        return jnp.concatenate([
            data.qpos,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ])


''' Register the environments to allow `env = envs.get_environment('double_integrator')` '''
envs.register_environment('double_integrator', DoubleIntegrator)



if __name__ == "__main__":
    
    
    env = envs.get_environment('double_integrator')
    env.reset(jax.random.key(0))
    print() 