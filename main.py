
import sys, os.path as osp
sys.path.append(osp.dirname(osp.abspath("")))

import jax, jax.numpy as jnp, numpy as np
import matplotlib.pyplot as plt

from utils import get_repo_root
from data_gen import *
from rom import *


if __name__ == "__main__":
    
    cfg_train = CfgTrain()
    cfg_loss = CfgLoss()

    dataset = DoubinteDataset(cfg=CfgDataLoad(), 
                            data_path=osp.join(get_repo_root(), "data/doubinte_data_5e4.npz"))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=DoubinteDataset.collate_fn)


    rom = NNDoubinteROM(CfgNNDoubinteROM())
    
    rom = train(rom, dataset, cfg_train, cfg_loss)
    
    post_train(rom, dataset, cfg_train, cfg_loss, f"{get_repo_root()}/logs/rom_5e4")

