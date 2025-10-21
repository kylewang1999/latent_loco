import jax, jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass
import flax.nnx as nnx
from typing import Callable
from typing import Tuple, Dict
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import jax_dataloader as jdl
import wandb
from einops import rearrange

from utils import catch_keyboard_interrupt, save_nnx_module
from data_gen import BaseDataset, IntegratorOutput


@flax_dataclass
class CfgROMBase: 
    nx: int|None = None
    nz: int|None = None
    nu: int|None = None

@flax_dataclass
class CfgNNDoubinteROM(CfgROMBase):
    nx: int = 2
    nz: int = 2
    nu: int = 1
    encoder_specs: Tuple[int, ...]|None = (16,16)
    decoder_specs: Tuple[int, ...]|None = (16,16)
    fz_specs: Tuple[int, ...] = (16,)
    act_fz: Callable = nnx.tanh
    act_encoder: Callable = nnx.tanh
    act_decoder: Callable = nnx.tanh
    use_residual_encoder: bool =True
    use_residual_decoder: bool =True
    use_residual_fz: bool = True
    
@flax_dataclass
class CfgNNCartPoleROM(CfgROMBase):
    nx: int = 4
    nz: int = 4
    nu: int = 1
    encoder_specs: Tuple[int, ...]|None = (16,16)
    decoder_specs: Tuple[int, ...]|None = (16,16)
    fz_specs: Tuple[int, ...] = (32,32,32)
    act_fz: Callable = nnx.tanh
    act_encoder: Callable = nnx.tanh
    act_decoder: Callable = nnx.tanh
    use_residual_encoder: bool = True
    use_residual_decoder: bool = True
    use_residual_fz: bool = True
    
@flax_dataclass
class CfgNNPaddleballROM(CfgROMBase):
    nx: int = 4
    nz: int = 4
    nu: int = 1
    nc: int = 1
    encoder_specs: Tuple[int, ...]|None = None
    decoder_specs: Tuple[int, ...]|None = None
    fz_specs: Tuple[int, ...] = tuple(256 for _ in range(6))
    act_fz: Callable = nnx.tanh
    act_encoder: Callable = nnx.gelu
    act_decoder: Callable = nnx.gelu
    use_residual_encoder: bool =True
    use_residual_decoder: bool =True
    use_residual_fz: bool = True


class MLP(nnx.Module):
    
    def __init__(self, din: int, dout: int, hidden_specs: Tuple[int, ...], 
                 act: Callable = nnx.tanh, use_residual: bool = True, *, rngs: nnx.Rngs):
        
        self.din = din
        self.dout = dout
        self.hidden_specs = hidden_specs
        self.act = act
        self.use_residual = use_residual
        
        # Build the main network layers
        self.layers = []
        self.projections = []  # For dimension matching in residual connections
        
        all_dims = [din] + list(hidden_specs) + [dout]
        
        for i in range(len(all_dims) - 1):
            din_layer = all_dims[i]
            dout_layer = all_dims[i + 1]
            
            # Main linear layer
            linear_layer = nnx.Linear(din_layer, dout_layer, use_bias=True, rngs=rngs)
            self.layers.append(linear_layer)
            
            # Projection layer for residual connection if dimensions don't match
            if self.use_residual and din_layer != dout_layer:
                proj_layer = nnx.Linear(din_layer, dout_layer, 
                                        use_bias=False, 
                                        kernel_init=nnx.initializers.orthogonal(),
                                        rngs=rngs)
                self.projections.append(proj_layer)
            else:
                self.projections.append(None)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        h = x
        for i, (layer, projection) in enumerate(zip(self.layers, self.projections)):
            h_input = h
            h = layer(h)
            
            if i < len(self.layers) - 1:
                h = self.act(h)

            if self.use_residual and i < len(self.layers) - 1:
                if projection is not None:
                    h = h + projection(h_input)
                else:
                    h = h + h_input
        
        return h
    
    
class Identity(nnx.Module):
    def __call__(self, x): return x


class BaseROM():
    
    def __init__(self, cfg: CfgROMBase):
        self.cfg = cfg

    def encode(self, x: jnp.ndarray) -> jnp.ndarray: raise NotImplementedError
    def decode(self, z: jnp.ndarray) -> jnp.ndarray: raise NotImplementedError
    def fz(self, z: jnp.ndarray) -> jnp.ndarray: raise NotImplementedError
    
    def loss_recon(self, batch:dict, *args, **kwargs) -> jnp.ndarray:
        x = batch['from'] # (b, nx)
        x_recon = self.decode(self.encode(x))
        diff = jnp.linalg.norm(x_recon - x, axis=-1)
        return jnp.mean(diff)
    
    def loss_reproj(self, batch:dict, *args, **kwargs) -> jnp.ndarray:
        x = batch['from'] # (b, nx)
        z = self.encode(x) # (b, nz)
        x_recon = self.decode(z)
        z_reproj = self.encode(x_recon)
        diff = jnp.linalg.norm(z_reproj - z, axis=-1)
        return jnp.mean(diff)
    
    def loss_fwd(self, batch:dict, pred_horizon: int|None = None) -> jnp.ndarray:
        x, xs_next, us = batch['from'], batch['to'], batch['ctrl']  # (b,nx), (b,pred_horizon,nx), (b,pred_horizon,nu)
        B, _pred_horizon, nu = us.shape
        T = _pred_horizon if pred_horizon is None else min(pred_horizon, _pred_horizon)
            
        z0 = self.encode(x)  # (B, nz)

        def step(carry, inputs):
            z, ut = carry, inputs
            z_next = self.fz(z, ut)   # (B, nz)
            return z_next, z_next     # carry zt, collect z_t

        us_TBN = jnp.swapaxes(us[:, :T], 0, 1)      # (T, B, nu)
        _, zs_pred_TBN = jax.lax.scan(step, z0, us_TBN)  # (T, B, nz)
        zs_pred = jnp.swapaxes(zs_pred_TBN, 0, 1)        # (B, T, nz)
        zs_true = self.encode(xs_next[:, :T])            # (B, T, nz)
        diff = jnp.linalg.norm(zs_pred - zs_true, axis=-1)  # (B, T)
        return diff.mean()
    
    def loss_bwd(self, batch:dict, pred_horizon: int|None = None) -> jnp.ndarray: 
        x, xs_next, us = batch['from'], batch['to'], batch['ctrl']
        B, _pred_horizon, nu = us.shape
        T = _pred_horizon if pred_horizon is None else min(pred_horizon, _pred_horizon)
        
        z0 = self.encode(x)  # (B, nz)

        def step(carry, inputs):
            z, ut = carry, inputs
            z_next = self.fz(z, ut)   # (B, nz)
            return z_next, z_next     # carry zt, collect z_t

        us_TBN = jnp.swapaxes(us[:, :T], 0, 1)           # (T, B, nu)
        _, zs_pred_TBN = jax.lax.scan(step, z0, us_TBN)  # (T, B, nz)
        zs_pred = jnp.swapaxes(zs_pred_TBN, 0, 1)        # (B, T, nz)
        xs_next_pred = self.decode(zs_pred)              # (B, T, nx)

        diff = jnp.linalg.norm(xs_next_pred - xs_next[:, :T], axis=-1)  # (B, T)
        return diff.mean()

 
class NNROM(BaseROM, nnx.Module):
    
    def __init__(self, cfg: CfgROMBase, *, rngs: nnx.Rngs = nnx.Rngs(0)):
        
        BaseROM.__init__(self, cfg)
        nnx.Module.__init__(self)
        
        nx, nz, nu = self.cfg.nx, self.cfg.nz, self.cfg.nu
        self.rngs = rngs

        self.nn_encoder = Identity() if self.cfg.encoder_specs is None else\
                          MLP(nx, nz, self.cfg.encoder_specs, act=self.cfg.act_encoder, 
                              use_residual=self.cfg.use_residual_encoder, rngs=self.rngs)

        self.nn_decoder = Identity() if self.cfg.decoder_specs is None else\
                          MLP(nz, nx, self.cfg.decoder_specs, act=self.cfg.act_decoder,
                              use_residual=self.cfg.use_residual_decoder, rngs=self.rngs)
            
        self.nn_fz = MLP(nz+nu, nz, self.cfg.fz_specs, act=self.cfg.act_fz,
                         use_residual=self.cfg.use_residual_fz, rngs=self.rngs)

        
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
    reproj: float = 0.0
    fwd: float = 1.0
    bwd: float = 1.0


@flax_dataclass
class CfgTrain:
    lr: float = 1e-2
    batch_size: int = 4096
    batch_size_eval: int = 256
    num_epochs: int = 10
    num_logs: int = 10
    ae_warmup_portion: float = 0.2   # percentage of total training steps (out of batch * epochs) for ae warmup
    max_train_pred_horizon: int = 4
    max_eval_pred_horizon: int = 4
    enable_lr_schedule: bool = True
    enable_grad_clipping: bool = False
    grad_clipping_value: float = 5.0
    train_portion: float = 0.9
    # wandb config # TODO: Implement wandb support
    use_wandb: bool = False
    wandb_project: str = "latent_loco"
    wandb_run_name: str|None = None
    wandb_log_frequency: int = 1  # log every N batches
    
    num_eval_long_rollout_traj: int = 100
    eval_rng_seed: int = 0
    


@catch_keyboard_interrupt("Training interrupted by user")
def train(rom: nnx.Module, dataloader: jdl.DataLoader,
          cfg_train: CfgTrain, cfg_loss: CfgLoss, collate_fn: Callable=lambda x: x):


    
    total_steps = cfg_train.num_epochs * len(dataloader)
    ae_warmup_steps = int(cfg_train.ae_warmup_portion * total_steps)
    rl_decay_steps = total_steps - ae_warmup_steps
    
    pred_horizon_schedule = optax.linear_schedule(init_value=1,
                                                  end_value=cfg_train.max_train_pred_horizon,
                                                  transition_steps=rl_decay_steps)
    if cfg_train.enable_lr_schedule:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.1 * cfg_train.lr, 
            peak_value=cfg_train.lr,
            warmup_steps=int(0.1 * rl_decay_steps), 
            decay_steps=rl_decay_steps,
        )
    else:
        lr_schedule = cfg_train.lr
    
    def make_tx() -> optax.GradientTransformation:
        transforms = [optax.adamw(lr_schedule)]
        if cfg_train.enable_grad_clipping:
            transforms.append(
                optax.adaptive_grad_clip(clipping=cfg_train.grad_clipping_value, eps=1e-3)
            )
        return optax.chain(*transforms)
    
    tx = make_tx()
    opt = nnx.Optimizer(rom, tx, wrt=nnx.Param)
    
    @nnx.jit(static_argnums=(3,4))
    def step(model: nnx.Module, opt: nnx.Optimizer, batch: dict, 
             in_ae_warmup: bool, pred_horizon: int):
        
        def loss_fn(m: BaseROM):
            
            loss_dict = {
                'total': jnp.asarray(0.0),
                'recon': jnp.asarray(0.0),
                'reproj': jnp.asarray(0.0),
                'fwd': jnp.asarray(0.0),
                'bwd': jnp.asarray(0.0),
            }
            
            for key in ['recon', 'reproj', 'fwd', 'bwd']:
                _l = getattr(m, f'loss_{key}')(batch)
                _l = jnp.nan_to_num(_l, nan=0.0) # replace nan with 0
                loss_dict[key] = _l
            
            total = jnp.sum(jnp.stack(list(loss_dict.values())))
            return total, loss_dict
        
        def loss_fn_old(m: BaseROM):
            recon = m.loss_recon(batch)
            
            if cfg_loss.reproj > 0:
                reproj = m.loss_reproj(batch)
            else:
                reproj = jnp.asarray(0.0)
            
            if cfg_loss.fwd > 0 and not in_ae_warmup:
                fwd = m.loss_fwd(batch, pred_horizon)
            else:
                fwd = jnp.asarray(0.0)
            
            if cfg_loss.bwd > 0 and not in_ae_warmup:
                bwd = m.loss_bwd(batch, pred_horizon)
            else:
                bwd = jnp.asarray(0.0)
            
            
            total = (cfg_loss.recon * recon
                     + cfg_loss.reproj * reproj
                     + cfg_loss.fwd * fwd
                     + cfg_loss.bwd * bwd)
            
            aux = {'total': total, 'recon': recon, 'reproj': reproj, 
                   'fwd': fwd, 'bwd': bwd,
                   'pred_horizon': pred_horizon}
            return total, aux

        (loss_val, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        opt.update(grads=grads)
        
        return model, opt, loss_val, aux


    global_step = 0
    epoch_losses = []
    for epoch in (pbar := tqdm(range(cfg_train.num_epochs))):
        
        batch_losses = []
        for i, batch in enumerate(dataloader):
            batch = collate_fn(batch)
            pred_horizon = int(pred_horizon_schedule(global_step - ae_warmup_steps) if global_step >= ae_warmup_steps else 1)
            
            in_ae_warmup = global_step < ae_warmup_steps
            rom, opt, loss, aux = step(rom, opt, batch, in_ae_warmup, pred_horizon)
            

            batch_losses.append(loss)
            global_step += 1
            
            pbar.set_postfix({
                "pred_horizon": f"{float(pred_horizon):.0f}",
                "b_loss": f"{float(loss):.2e}", 
                "b_prog": f"{i}/{len(dataloader)}",
                "in_ae_warmup": in_ae_warmup,
            })
        
        # log epoch-level metrics
    
        epoch_loss = jnp.mean(jnp.stack(batch_losses))
        epoch_losses.append(epoch_loss)
        lr_val = (lr_schedule(global_step) if cfg_train.enable_lr_schedule
                    else jnp.asarray(cfg_train.lr))
        pbar.set_description(f"Loss: {float(epoch_loss):.2e}, LR: {float(lr_val):.2e}")
        
        # Log epoch-level metrics to wandb
        if cfg_train.use_wandb:
            wandb.log({
                "train/epoch_loss": float(epoch_loss),
                "train/epoch": epoch
            }, step=global_step)

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(jnp.arange(len(epoch_losses)), epoch_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    plt.show()
    
    # Finish wandb run
    if cfg_train.use_wandb:
        wandb.finish()
    
    return rom


def evaluate_cts(rom: NNROM):
    

    q0s, v0s = jnp.meshgrid(jnp.linspace(0, 4, n0:=20), 
                            jnp.linspace(-8, 5, n2:=20))
    q1s, v1s = jnp.meshgrid(jnp.linspace(-0.5, 1, n1:=20), 
                            jnp.linspace(-5,5, n3:=20))
    xs_query_02 = jnp.stack([q0s, jnp.zeros_like(q0s), v0s, jnp.zeros_like(v0s)], axis=-1)
    xs_query_13 = jnp.stack([jnp.zeros_like(q1s), q1s, jnp.zeros_like(v1s), v1s], axis=-1)
    zs_query_02 = rom.encode(xs_query_02)
    zs_query_13 = rom.encode(xs_query_13)
    
    dzs_query_02 = rom.fz(rearrange(zs_query_02, "nx ny nz -> (nx ny) nz"), jnp.zeros((n0*n2, 1)))
    dzs_query_02 = rearrange(dzs_query_02, "(nx ny) nz -> nx ny nz", nx=n0, ny=n2)
    dzs_query_02 -= zs_query_02
    
    dzs_query_13 = rom.fz(rearrange(zs_query_13, "nx ny nz -> (nx ny) nz"), jnp.ones((n1*n3, 1)))
    dzs_query_13 = rearrange(dzs_query_13, "(nx ny) nz -> nx ny nz", nx=n0, ny=n2)
    dzs_query_13 -= zs_query_13

    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    axes[0][0].set_title("q0, v0; ground truth")
    axes[0][1].set_title("q0, v0; predicted")
    axes[1][0].set_title("q1, v1; ground truth")
    axes[1][1].set_title("q1, v1; predicted")
    
    axes[0][1].quiver(q0s, v0s, dzs_query_02[:,:,0], dzs_query_02[:,:,2])
    axes[1][1].quiver(q1s, v1s, dzs_query_13[:,:,1], dzs_query_13[:,:,3])
    plt.show()

    

def evaluate(rom: nnx.Module, dataset: Dataset, cfg_train: CfgTrain, cfg_loss: CfgLoss):
    '''
    Parameters
    ----------
    rom: nnx.Module
        The ROM model to evaluate.
    dataset: Dataset
        The dataset to evaluate on. This is a subset object, not the BaseDataset object.
    cfg_train: CfgTrain
    cfg_loss: CfgLoss
    '''
    
    dataloader = jdl.DataLoader(dataset, batch_size=cfg_train.batch_size_eval, 
                                backend='pytorch', shuffle=False)

    batch = next(iter(dataloader))
    batch = dataset.collate_fn(batch)
    x, xs_next, us = batch['from'], batch['to'], batch['ctrl']  # (b,nx), (b,pred_horizon,nx), (b,pred_horizon,nu)
    print(f"x.shape: {x.shape}, xs_next.shape: {xs_next.shape}, us.shape: {us.shape}")
    
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

    fig1, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].set_title(r"Initial States $x_0$")
    axes[1].set_title(r"Reconstructed States $D(E(x_0))$")
    axes[2].set_title(r"Encoded States $E(x_0)$")
    axes[3].set_title(r"Re-Encoded States $E(D(E(x_0)))$")
    for (ax, data) in zip(axes, [x, x_recon, z, z_reproj]):
        for i in range(data.shape[0]):
            ax.scatter(data[i,0], data[i,1])
    plt.show()
    
    fig2, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].set_title(r"Rollout $E(x_{t:t+\tau})$, $\tau=$" + str(pred_horizon))
    axes[1].set_title(r"Rollout Pred $E(x_t) + f_z^{(\tau)}\circ E(x_t)$")
    axes[2].set_title(r"Rollout $x_{t:t+\tau+1}$, $\tau=$" + str(pred_horizon))
    axes[3].set_title(r"Rollout Pred $x_t + D(f_z^{(\tau)}\circ E(x_t))$")
    
    
    # axes[1].set_xlim(axes[0].get_xlim())
    # axes[1].set_ylim(axes[0].get_ylim())
    # axes[3].set_xlim(axes[2].get_xlim())
    # axes[3].set_ylim(axes[2].get_ylim())
    
    
    for (ax, data) in zip(axes, [zs_next, zs_next_pred, xs_next, xs_next_pred]):
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        for i in range(data.shape[0]):
            ax.plot(data[i,:,0], data[i,:,1])
            ax.plot(data[i,0,0], data[i,0,1], 'r.')
            ax.plot(data[i,-1,0], data[i,-1,1], 'g*')
    plt.show()
    
    
def evaluate_long_rollout(rom: nnx.Module, dataset: BaseDataset, cfg_train: CfgTrain, cfg_loss: CfgLoss):
    '''
    Parameters
    ----------
    rom: nnx.Module
        The ROM model to evaluate.
    dataset: BaseDataset
        The dataset to evaluate on,
    cfg_train: CfgTrain
    cfg_loss: CfgLoss
    '''
    
    data:IntegratorOutput = dataset.data
    N, T = data.num_traj, data.num_tsteps-1
    B = cfg_train.num_eval_long_rollout_traj
    pick_inds = jax.random.choice(jax.random.PRNGKey(cfg_train.eval_rng_seed), N, (B,))
    
    xs = jnp.asarray(data.xs[pick_inds])
    zs = rom.encode(xs)
    us = jnp.asarray(data.us[pick_inds])
    cs = None if data.cs is None else jnp.asarray(data.cs[pick_inds])

    z = zs[:,0,:]
    x_recon = rom.decode(z)
    z_reproj = rom.encode(x_recon)
    zs_next_pred = jnp.zeros((B, T+1, rom.cfg.nz))
    xs_next_pred = jnp.zeros((B, T+1, rom.cfg.nx))
    
    def step(t, carry):
        z, zs_next_pred, xs_next_pred = carry
        z = rom.fz(z, us[:,t])
        x_dec = rom.decode(z)
        zs_next_pred = zs_next_pred.at[:,t].set(z)
        xs_next_pred = xs_next_pred.at[:,t].set(x_dec)
        return z, zs_next_pred, xs_next_pred
    
    _, zs_next_pred, xs_next_pred = jax.lax.fori_loop(0, T, step, (z, zs_next_pred, xs_next_pred))
    
    state_annotations = ["q0", "q1", "v0", "v1"]
    ts = jnp.arange(T+1)
    fig, axes = plt.subplots(4, 2, figsize=(10,15))
    for i in range(len(axes)):
        axes[i][0].set_title(f"{state_annotations[i]}, ground truth")
        axes[i][1].set_title(f"{state_annotations[i]}, predicted")
        for j in range(B):
            axes[i][0].plot(ts, xs[j,:,i])
            axes[i][1].plot(ts, xs_next_pred[j,:,i])
    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    state_annotations = ['q0 vs v0', 'q1 vs v1']
    for i in range(len(axes)):
        axes[i][0].set_title(f"{state_annotations[i]}, ground truth")
        axes[i][1].set_title(f"{state_annotations[i]}, predicted")
    
    for j in range(B):
        axes[0][0].plot(xs[j,:,0], xs[j,:,2], )
        axes[0][1].plot(xs_next_pred[j,:,0], xs_next_pred[j,:,2])
        axes[1][0].plot(xs[j,:,1], xs[j,:,3])
        axes[1][1].plot(xs_next_pred[j,:,1], xs_next_pred[j,:,3])
    plt.show()
    
    
    # dzs = zs_next_pred[:,:-1,:] - zs_next_pred[:,1:,:]
    # print(dzs.shape)
    
    grid_q0 = jnp.linspace(0, 4, n0:=20)
    grid_v0 = jnp.linspace(-8, 5, n2:=20)
    grid_q1 = jnp.linspace(-0.5, 1, n1:=20)
    grid_v1 = jnp.linspace(-5,5, n3:=20)
    q0s, v0s = jnp.meshgrid(grid_q0, grid_v0)
    q1s, v1s = jnp.meshgrid(grid_q1, grid_v1)
    xs_query_02 = jnp.stack([q0s, jnp.zeros_like(q0s), v0s, jnp.zeros_like(v0s)], axis=-1)
    xs_query_13 = jnp.stack([jnp.zeros_like(q1s), q1s, jnp.zeros_like(v1s), v1s], axis=-1)
    zs_query_02 = rom.encode(xs_query_02)
    zs_query_13 = rom.encode(xs_query_13)
    
    dzs_query_02 = rom.fz(rearrange(zs_query_02, "nx ny nz -> (nx ny) nz"), jnp.zeros((n0*n2, 1)))
    dzs_query_02 = rearrange(dzs_query_02, "(nx ny) nz -> nx ny nz", nx=n0, ny=n2)
    dzs_query_02 -= zs_query_02
    
    dzs_query_13 = rom.fz(rearrange(zs_query_13, "nx ny nz -> (nx ny) nz"), jnp.ones((n0*n2, 1)))
    dzs_query_13 = rearrange(dzs_query_13, "(nx ny) nz -> nx ny nz", nx=n0, ny=n2)
    dzs_query_13 -= zs_query_13

    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    axes[0][0].set_title("q0, v0; ground truth")
    axes[0][1].set_title("q0, v0; predicted")
    axes[1][0].set_title("q1, v1; ground truth")
    axes[1][1].set_title("q1, v1; predicted")
    
    axes[0][1].quiver(q0s, v0s, dzs_query_02[:,:,0], dzs_query_02[:,:,2])
    axes[1][1].quiver(q1s, v1s, dzs_query_13[:,:,1], dzs_query_13[:,:,3])
    plt.show()
    


def post_train(rom: nnx.Module, dataset: Dataset, 
               cfg_train: CfgTrain, cfg_loss: CfgLoss, save_dir: str):
    
    save_nnx_module(rom, save_dir)


if __name__ == "__main__":
    
    pass