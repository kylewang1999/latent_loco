import os, os.path as osp
import jax, jax.numpy as jnp, numpy as np
import matplotlib.pyplot as plt
from jax import vmap, jit, grad
from jax.tree_util import tree_map
from flax.struct import dataclass as flax_dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import Callable
from utils import timeit, CONSOLE_LOGGER, get_repo_root

def rk4(f, x, u, dt):
    ''' f should take as input (x, u) and return (x_dot, u_dot), unbatched. '''
    k1 = f(x          , u) 
    k2 = f(x + dt*k1/2, u)
    k3 = f(x + dt*k2/2, u)
    k4 = f(x + dt*k3  , u) 
    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_batch(f, xs, us, dt):
    ''' f should take as input (x, u) and return (x_dot, u_dot), unbatched. '''
    f = vmap(f, in_axes=(0,0))
    k1 = f(xs, us)
    k2 = f(xs + dt*k1/2, us)
    k3 = f(xs + dt*k2/2, us)
    k4 = f(xs + dt*k3  , us)
    return xs + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


''' Dynamics, Dataset '''

@flax_dataclass
class IntegratorOutput:
    xs: jnp.ndarray
    us: jnp.ndarray
    
    @property
    def attrs(self): return ("xs", "us")

    
@dataclass
class CfgDataGen:
    nx: int = 2
    nu: int = 1
    rand_u_scale: float = 10.0
    dt: float = 0.02
    t_max: int = 0.2
    ntraj: int = int(5e4)
    
    @property
    def n_steps(self): return int(self.t_max / self.dt)

    
class DoubleIntegrator:
    
    def __init__(self, cfg: CfgDataGen, rng: jnp.ndarray=jax.random.PRNGKey(0)):
        self.cfg = cfg
        self.rng = rng


    def f(self, x, u):
        x1dot = x[1:2]
        x2dot = jnp.atleast_1d(u)
        return jnp.hstack([x1dot, x2dot])  # (2,)
    
    
    def policy_random_batch(self, xs, key):
        key, subkey = jax.random.split(key)
        us = jax.random.normal(subkey, (xs.shape[0], 1)) * self.cfg.rand_u_scale
        return us, key
    
    @timeit
    def rollout(self, x0s, dt, fn_policy_batch:Callable=None):
        
        ''' fn_policy_batch should take as input (xs) and return (us), batched. 
        Returns IntegratorOutput object.
            xs: (B, T+1, nx)
            us: (B, T, nu)
        '''
        if fn_policy_batch is None:
            fn_policy_batch = self.policy_random_batch
            
        key = self.rng
        
        def step(i, carry):
            inte_out, key = carry
            xs_curr = inte_out.xs[:,i]
            us_curr, key = fn_policy_batch(xs_curr, key)
            x_next = rk4_batch(self.f, xs_curr, us_curr, dt)
            
            inte_out = inte_out.replace(xs=inte_out.xs.at[:,i+1].set(x_next),
                                        us=inte_out.us.at[:,i].set(us_curr))
            return inte_out, key
        
        inte_out = IntegratorOutput(jnp.zeros((x0s.shape[0], self.cfg.n_steps+1, self.cfg.nx)),
                                    jnp.zeros((x0s.shape[0], self.cfg.n_steps, self.cfg.nu)))
        inte_out = inte_out.replace(xs=inte_out.xs.at[:,0].set(x0s))
        inte_out, _ = jax.lax.fori_loop(0, self.cfg.n_steps, step, (inte_out, key))
        
        return inte_out
        

''' IO '''
def save_rollout(path:str, inte_out:IntegratorOutput) -> None:
    xs = np.asarray(inte_out.xs).astype(np.float32)
    us = np.asarray(inte_out.us).astype(np.float32)
    np.savez(path, xs=xs, us=us)
    CONSOLE_LOGGER.info(f"Rollout saved to {path} | Size: {(xs.nbytes + us.nbytes) / (1024**3):.2f} GB")
    
def load_rollout(path:str) -> IntegratorOutput:
    data = np.load(path)
    CONSOLE_LOGGER.info(f"Rollout loaded from {path} | Size: {(data['xs'].nbytes + data['us'].nbytes) / (1024**3):.2f} GB")
    return IntegratorOutput(xs=data['xs'], us=data['us'])



class BaseDataset(Dataset):
    
    def __init__(self, pred_horizon: int, data_path:str):
                
        ''' Datset for multistep dynamics prediction learning 
        Converts [xs (B, T+1, nx), us (B, T, nu)] to batches of the form:
        {
            'from': init states (b, 1, nx),
            'ctrl': controls (b, pred_horizon, nu),
            'to':   goal states (b, pred_horizon, nx),
        }
        '''
        
        self.pred_horizon = pred_horizon
        self.data_path = data_path
        self.data: IntegratorOutput|None = None
        self.max_tstep: int|None = None
        self.window_start_inds: list[tuple[int, int]]|None = None
        
    
    def __len__(self): return len(self.window_start_inds)
    
    
    def __getitem__(self, idx):
        i, t = self.window_start_inds[idx]
        
        from_ = self.data.xs[i, t:t+1].squeeze(0)  # (1, nx) -> (nx,)
        ctrl_ = self.data.us[i, t:t+self.pred_horizon]  # (pred_horizon, nu)
        to_ = self.data.xs[i, t+1:t+self.pred_horizon+1]  # (pred_horizon, nx)
        
        return {'from': from_, 'ctrl': ctrl_, 'to': to_}
    
    
    def _log(self):
        CONSOLE_LOGGER.info(f"DoubinteDataset initialized with {len(self.window_start_inds)} windows/items")
        CONSOLE_LOGGER.info(f"\t pred_horizon: {self.pred_horizon}, max_tstep: {self.max_tstep}")
        CONSOLE_LOGGER.info(f"\t xs_raw_shape: {self.data.xs.shape}, us_raw_shape: {self.data.us.shape}")
        


    @staticmethod
    def collate_fn(batch):
        batch = tree_map(jnp.asarray, batch)
        return batch


class DoubinteDataset(BaseDataset):
    
    def __init__(self, pred_horizon: int, data_path:str=osp.join(get_repo_root(), "data/doubinte_data_5e4.npz")):
        
        super().__init__(pred_horizon, data_path)
        
        self.pred_horizon = pred_horizon
        self.data_path = data_path
        self.data: IntegratorOutput = load_rollout(data_path)
        
        self.max_tstep = self.data.us.shape[1]
        
        self.window_start_inds = [(i, t) for i in range(self.data.us.shape[0]) 
                                         for t in range(self.max_tstep - self.pred_horizon - 1)]
        
        super()._log()
        



class CartPoleDataset(BaseDataset):
    
    def __init__(self, pred_horizon: int, data_path:str=osp.join(get_repo_root(), "data/cart_pole_data.npz")):
        
        super().__init__(pred_horizon, data_path)
        
        self.pred_horizon = pred_horizon
        self.data_path = data_path
        
        data_np = np.load(data_path)
        
        self.data:IntegratorOutput = IntegratorOutput(xs=np.concatenate([data_np['q_traj'], data_np['v_traj']], axis=-1), 
                                                      us=data_np['u_traj'])
        
        self.max_tstep = self.data.us.shape[1]
        
        self.window_start_inds = [(i, t) for i in range(self.data.us.shape[0]) 
                                         for t in range(self.max_tstep - self.pred_horizon - 1)]
        
        super()._log()



if __name__ == "__main__":
    cfg = CfgDataGen()
    di = DoubleIntegrator(cfg)
    x0s = jnp.zeros((14, 2))
    inte_out = di.rollout(x0s, cfg.dt)
    print(inte_out.xs.shape, inte_out.us.shape)