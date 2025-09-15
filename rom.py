import jax, jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass
import flax.nnx as nnx
from typing import Callable
from typing import Tuple, Dict
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from utils import catch_keyboard_interrupt, save_nnx_module


@flax_dataclass
class CfgROMBase: 
    nx: int|None = None
    nz: int|None = None
    nu: int|None = None


class MLP(nnx.Module):
    
    def __init__(self, din: int, dout: int, hidden_specs: Tuple[int, ...], 
                 act: Callable = nnx.tanh, *, rngs: nnx.Rngs):
        
        layers = []
        dprev = din
        for w in hidden_specs:
            layers.append(nnx.Linear(dprev, w, use_bias=True, rngs=rngs))
            layers.append(act)
            dprev = w
        layers.append(nnx.Linear(dprev, dout, use_bias=True, rngs=rngs))
        
        self.net = nnx.Sequential(*layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.net(x)
    
    
class Identity(nnx.Module):
    def __call__(self, x): return x


class BaseROM():
    
    def __init__(self, cfg: CfgROMBase):
        self.cfg = cfg

    def encode(self, x: jnp.ndarray) -> jnp.ndarray: raise NotImplementedError
    def decode(self, z: jnp.ndarray) -> jnp.ndarray: raise NotImplementedError
    def fz(self, z: jnp.ndarray) -> jnp.ndarray: raise NotImplementedError
    
    def loss_recon(self, batch:dict) -> jnp.ndarray:
        x = batch['from'] # (b, nx)
        x_recon = self.decode(self.encode(x))
        diff = jnp.linalg.norm(x_recon - x, axis=-1)
        return jnp.mean(diff)
    
    def loss_reproj(self, batch:dict) -> jnp.ndarray:
        x = batch['from'] # (b, nx)
        z = self.encode(x) # (b, nz)
        x_recon = self.decode(z)
        z_reproj = self.encode(x_recon)
        diff = jnp.linalg.norm(z_reproj - z, axis=-1)
        return jnp.mean(diff)
    
    def loss_fwd(self, batch:dict, pred_horizon: int|None = None) -> jnp.ndarray:
        x, xs_next, us = batch['from'], batch['to'], batch['ctrl']  # (b,nx), (b,pred_horizon,nx), (b,pred_horizon,nu)
        B, _pred_horizon, nu = us.shape
        pred_horizon = _pred_horizon if pred_horizon is None else min(pred_horizon, _pred_horizon)
        
        z = self.encode(x)
        zs_next_pred = jnp.zeros((B, pred_horizon, self.cfg.nz))
        
        def step(t, carry):
            z, zs_next_pred = carry
            z = self.fz(z, us[:,t])
            zs_next_pred = zs_next_pred.at[:,t].set(z)
            return z, zs_next_pred
        
        _, zs_next_pred = jax.lax.fori_loop(0, pred_horizon, step, (z, zs_next_pred))   
        zs_next = self.encode(xs_next)
        diff = jnp.linalg.norm(zs_next_pred - zs_next, axis=-1)
        return jnp.mean(diff)
    
    def loss_bwd(self, batch:dict, pred_horizon: int|None = None) -> jnp.ndarray: 
        x, xs_next, us = batch['from'], batch['to'], batch['ctrl']
        B, _pred_horizon, nu = us.shape
        pred_horizon = _pred_horizon if pred_horizon is None else min(pred_horizon, _pred_horizon)
        
        z = self.encode(x)
        xs_next_pred = jnp.zeros((B, pred_horizon, self.cfg.nx))
        
        def step(t, carry):
            z, xs_pred = carry
            z = self.fz(z, us[:,t])
            x_dec = self.decode(z)
            xs_pred = xs_pred.at[:, t].set(x_dec)
            return (z, xs_pred)

        _, xs_next_pred = jax.lax.fori_loop(0, pred_horizon, step, (z, xs_next_pred))
        diff = jnp.linalg.norm(xs_next_pred - xs_next, axis=-1)  # (B,T)
        return jnp.mean(diff)

 
@flax_dataclass
class CfgNNDoubinteROM(CfgROMBase):
    nx: int = 2
    nz: int = 2
    nu: int = 1
    encoder_specs: Tuple[int, ...]|None = None
    decoder_specs: Tuple[int, ...]|None = None
    fz_specs: Tuple[int, ...] = (16,)
    act: Callable = nnx.tanh
    

class NNDoubinteROM(BaseROM, nnx.Module):
    
    def __init__(self, cfg: CfgNNDoubinteROM, *, rngs: nnx.Rngs = nnx.Rngs(0)):
        
        BaseROM.__init__(self, cfg)
        nnx.Module.__init__(self)
        
        nx, nz, nu = self.cfg.nx, self.cfg.nz, self.cfg.nu
        self.rngs = rngs

        self.nn_encoder = Identity() if self.cfg.encoder_specs is None else\
                          MLP(nx, nz, self.cfg.encoder_specs, act=self.cfg.act, rngs=self.rngs)

        self.nn_decoder = Identity() if self.cfg.decoder_specs is None else\
                          MLP(nz, nx, self.cfg.decoder_specs, act=self.cfg.act, rngs=self.rngs)
            
        print(nz, nu, nz+nu)
        self.nn_fz = MLP(nz+nu, nz, self.cfg.fz_specs, act=self.cfg.act, rngs=self.rngs)

        
    def encode(self, x: jnp.ndarray) -> jnp.ndarray:  # natively batched, but vmap also supported
        return self.nn_encoder(x)
    
    def decode(self, z: jnp.ndarray) -> jnp.ndarray:  # natively batched, but vmap also supported
        return self.nn_decoder(z)
    
    def fz(self, z: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:  # natively batched, but vmap also supported
        return self.nn_fz(jnp.hstack([z, u])) + z

    def loss_recon(self, *args, **kwargs) -> jnp.ndarray: return super().loss_recon(*args, **kwargs)
    def loss_reproj(self, *args, **kwargs) -> jnp.ndarray: return super().loss_reproj(*args, **kwargs)
    def loss_fwd(self, *args, **kwargs) -> jnp.ndarray: return super().loss_fwd(*args, **kwargs)
    def loss_bwd(self, *args, **kwargs) -> jnp.ndarray: return super().loss_bwd(*args, **kwargs)


class LossState(nnx.TrainState):
    recon: jnp.ndarray
    reproj: jnp.ndarray
    fwd: jnp.ndarray
    bwd: jnp.ndarray
    total: jnp.ndarray
    
    
def make_loss_plots(inte_out: Dict,
                    loss_out: LossState, 
                    title: str = "Losses") -> plt.Figure:
    
    fig, axes = plt.subplots(2, len(loss_out.attrs)//2, figsize=(4*len(loss_out.attrs), 5))
    axes = axes.flatten()
    fig.suptitle(title)
    for ax, attr in zip(axes, loss_out.attrs):
        ax.set_title(attr)
        ax.set_xlabel('t')
        ax.grid(True, alpha=0.3)
    
        raise NotImplementedError

    plt.show()
    return fig


@flax_dataclass
class CfgLoss:
    recon: float = 1.0
    reproj: float = 1.0
    fwd: float = 1.0
    bwd: float = 1.0


@flax_dataclass
class CfgTrain:
    lr: float = 5e-2
    batch_size: int = 4096
    batch_size_eval: int = 256
    num_epochs: int = 10
    num_logs: int = 10
    max_train_pred_horizon: int = 4
    max_eval_pred_horizon: int = 4
    enable_lr_schedule: bool = False
    enable_grad_clipping: bool = True
    grad_clipping_value: float = 5.0
    train_portion: float = 0.9


@catch_keyboard_interrupt("Training interrupted by user")
def train(rom: nnx.Module, dataset: Dataset,
          cfg_train: CfgTrain, cfg_loss: CfgLoss):
    
    dataloader = DataLoader(dataset, batch_size=cfg_train.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    total_steps = cfg_train.num_epochs * len(dataloader)
    if cfg_train.enable_lr_schedule:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.1 * cfg_train.lr, 
            peak_value=cfg_train.lr,
            warmup_steps=int(0.1 * total_steps), 
            decay_steps=total_steps,
        )
    else:
        lr_schedule = cfg_train.lr
    
    def make_tx() -> optax.GradientTransformation:
        transforms = [optax.adam(lr_schedule)]
        if cfg_train.enable_grad_clipping:
            transforms.append(
                optax.adaptive_grad_clip(clipping=cfg_train.grad_clipping_value,eps=1e-3)
            )
        return optax.chain(*transforms)
    
    tx = make_tx()
    opt = nnx.Optimizer(rom, tx, wrt=nnx.Param)
    
    @nnx.jit
    def step(model: nnx.Module, batch: dict, cfg_loss: CfgLoss):
        
        def loss_fn(m: BaseROM):
            recon = m.loss_recon(batch)
            reproj = m.loss_reproj(batch)
            fwd   = m.loss_fwd(batch, cfg_train.max_train_pred_horizon)
            bwd   = m.loss_bwd(batch, cfg_train.max_train_pred_horizon)
            total = (cfg_loss.recon * recon
                     + cfg_loss.reproj * reproj
                     + cfg_loss.fwd * fwd
                     + cfg_loss.bwd * bwd)
            aux = {'recon': recon, 'reproj': reproj, 'fwd': fwd, 'bwd': bwd, 'total': total}
            return total, aux

        (loss_val, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        return loss_val, aux, grads


    global_step = 0
    epoch_losses = []
    for epoch in (pbar := tqdm(range(cfg_train.num_epochs))):
        
        batch_losses = []
        for i, batch in enumerate(dataloader):
            loss, aux, grads = step(rom, batch, cfg_loss)
            opt.update(grads=grads)

            batch_losses.append(loss)
            global_step += 1
            pbar.set_postfix({"b_loss": f"{float(loss):.2e}", "b_progress": f"{i}/{len(dataloader)}"})

    
        epoch_loss = jnp.mean(jnp.stack(batch_losses))
        epoch_losses.append(epoch_loss)
        lr_val = (lr_schedule(global_step) if cfg_train.enable_lr_schedule
                    else jnp.asarray(cfg_train.lr))
        pbar.set_description(f"Loss: {float(epoch_loss):.2e}, LR: {float(lr_val):.2e}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(jnp.arange(len(epoch_losses)), epoch_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    plt.show()
    
    return rom


def evaluate(rom: nnx.Module, dataset: Dataset, cfg_train: CfgTrain, cfg_loss: CfgLoss):
    
    dataloader = DataLoader(dataset, batch_size=cfg_train.batch_size_eval, shuffle=False, collate_fn=dataset.collate_fn)

    batch = next(iter(dataloader))
    x, xs_next, us = batch['from'], batch['to'], batch['ctrl']  # (b,nx), (b,pred_horizon,nx), (b,pred_horizon,nu)
    B, _pred_horizon, nu = us.shape
    pred_horizon = cfg_train.max_eval_pred_horizon
    
    z = rom.encode(x)
    x_recon = rom.decode(z)
    z_reproj = rom.encode(x_recon)
    zs_next_pred = jnp.zeros((B, pred_horizon, rom.cfg.nz))
    xs_next_pred = jnp.zeros((B, pred_horizon, rom.cfg.nx))
    
    def step(t, carry):
        z, zs_next_pred, xs_next_pred = carry
        z = rom.fz(z, us[:,t])
        x_dec = rom.decode(z)
        zs_next_pred = zs_next_pred.at[:,t].set(z)
        xs_next_pred = xs_next_pred.at[:, t].set(x_dec)
        return z, zs_next_pred, xs_next_pred
    
    _, zs_next_pred, xs_next_pred = jax.lax.fori_loop(0, pred_horizon, step, (z, zs_next_pred, xs_next_pred))   
    zs_next = rom.encode(xs_next)

    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_title(r"Initial States $x_0$")
    axes[1].set_title(r"Reconstructed States $D(E(x_0))$")
    axes[2].set_title(r"Reprojected States $E(D(E(x_0)))$")
    for (ax, data) in zip(axes, [x, x_recon, z_reproj]):
        ax.scatter(data[:,0], data[:,1])
    plt.show()
    
    fig2, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].set_title(r"Rollout $E(x_{\tau+1})$, $\tau=$" + str(pred_horizon))
    axes[1].set_title(r"Rollout Pred $E(x_t) + f_z^{(\tau)}\circ E(x_t)$")
    axes[2].set_title(r"Rollout $x_{\tau+1}$, $\tau=$" + str(pred_horizon))
    axes[3].set_title(r"Rollout Pred $x_t + D(f_z^{(\tau)}\circ E(x_t))$")
    
    for (ax, data) in zip(axes, [zs_next, zs_next_pred, xs_next, xs_next_pred]):
        for i in range(data.shape[0]):
            ax.plot(data[i,:,0], data[i,:,1])
            ax.plot(data[i,0,0], data[i,0,1], 'r.')
            ax.plot(data[i,-1,0], data[i,-1,1], 'g*')
    plt.show()
    
    

def post_train(rom: nnx.Module, dataset: Dataset, 
               cfg_train: CfgTrain, cfg_loss: CfgLoss, save_dir: str):
    
    save_nnx_module(rom, save_dir)


if __name__ == "__main__":
    
    pass