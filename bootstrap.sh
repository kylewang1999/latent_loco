#! /bin/bash
# WARNING: Create and activate your conda environment before running this script
# To create and activate the environment, run:
# conda deactivate && conda remove -n latent_loco --all -y && conda create -n latent_loco python=3.10 -y && conda activate latent_loco && bash bootstrap.sh
# 
# Verify with: python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)"

pip install --upgrade pip

pip install mujoco pygame \
    mujoco-mjx brax flax orbax-checkpoint jax-dataloader\
    hydra-core einops\
    matplotlib imageio-ffmpeg \
    wandb ipython ipykernel tqdm rich nbformat mediapy 

# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu # just for dataloader - so cpu only
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # if you have want to use torch with cuda

pip install --upgrade "jax[cuda12]"  # wheels only available on linux.
pip install diffrax

# To fix jax CuSolver error: 
# https://github.com/jax-ml/jax/issues/29042, https://github.com/jax-ml/jax/issues/29065
pip install --upgrade nvidia-cublas-cu12==12.9.0.13

