import sys, os, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import matplotlib.pyplot as plt
from torch.utils.data import random_split

import jax, jax.numpy as jnp, numpy as np

from rom import (CfgNNDoubinteROM, CfgTrain, CfgLoss, 
                 NNROM, train, save_nnx_module, evaluate)
from utils import *
from data_gen import *



if __name__ == "__main__":

    
    ''' 1. Generate data for double integrator '''
    
    ntraj = "5e4"
    out_file_dir = osp.join(get_repo_root(), "data")
    out_file_path = osp.join(out_file_dir, f"doubinte_data_{ntraj}.npz")
    if not osp.exists(out_file_dir):
        os.makedirs(out_file_dir)
        
    if not osp.exists(out_file_path):

        cfg_datagen = CfgDataGen(ntraj=int(float(ntraj)))
        rng = jax.random.PRNGKey(1234)
        di = DoubleIntegrator(cfg_datagen, rng)
        x0s = jax.random.normal(rng, (cfg_datagen.ntraj, 2))

        inte_out = di.rollout(x0s, cfg_datagen.dt)
        save_rollout(out_file_path, inte_out)
    
    else:
        CONSOLE_LOGGER.info(f"Data already exists at {out_file_path}")
        
    
    ''' 2. Train ROM '''
    cfg_rom = CfgNNDoubinteROM(encoder_specs=None, decoder_specs=None)
    cfg_train = CfgTrain(num_epochs=50, batch_size=8192)
    cfg_loss = CfgLoss()
    dataset_size = "5e4"  # "5e4" or "5e7"
    rng_seed = 0

    dataset = DoubinteDataset(pred_horizon=cfg_train.max_train_pred_horizon, 
                            data_path=osp.join(get_repo_root(), f"data/doubinte_data_{dataset_size}.npz"))
    train_set, eval_set = random_split(dataset, [cfg_train.train_portion, 1-cfg_train.train_portion])
    train_set.collate_fn = DoubinteDataset.collate_fn
    eval_set.collate_fn = DoubinteDataset.collate_fn


    # NOTE: Modify the `train_from_scratch` flag to train from scratch or not.
    if train_from_scratch:=True:
        rom = NNROM(cfg_rom, rngs=nnx.Rngs(rng_seed))
        _ = train(rom, train_set, cfg_train, cfg_loss)
        save_nnx_module(rom, f"{get_repo_root()}/logs/rom_{dataset_size}")

    else:
        rom = restore_nnx_module(lambda: NNROM(cfg_rom, rngs=nnx.Rngs(rng_seed)),
                                f"{get_repo_root()}/logs/rom_{dataset_size}")

    evaluate(rom, eval_set, cfg_train, cfg_loss)