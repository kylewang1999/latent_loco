# Latent Dynamics Learning for Locomotion and RoMs

## 1. Environment Setup

The following setup procedure is meant for Linux system with NVIDIA CUDA GPU.

```bash
conda deactivate && conda remove -n latent_loco --all -y && conda create -n latent_loco python=3.10 -y && conda activate latent_loco && bash bootstrap.sh
python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)"
python -c "import torch; print(torch.cuda.is_available())"
```

Upstream repos:
- Paul: https://github.com/paullutkus/jax_latents/tree/main
- Surgio: https://github.com/sesteban951/LearningROMs


## 2. Examples

1. Double integrator dynamics learning. See [train_doubleinte.ipynb](scripts/train_doubleinte.ipynb) for more details; or run: `python scripts/train_doubleinte.py`

2. Cartpole dynamics learning. See [train_cartpole.ipynb](scripts/train_cartpole.ipynb) for more details.



## 3. Common Problems

1\. Jax cannot find cuda dependencies. 
Error message:
```
RuntimeError: Unable to load cuBLAS. Is it installed?
WARNING:2025-09-11 23:56:40,743:jax._src.xla_bridge:794: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
WARNING:jax._src.xla_bridge:An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```
Solution: Unset LD_LIBRARY_PATH for this shell session and jax xla should be able to find cuda dependencies.
```bash
unset LD_LIBRARY_PATH
python -c "import jax; print(jax.extend.backend.get_backend().platform)"
```

A more permanent solution is to add the following to your `~/.bashrc` file:
```bash
python()  { (unset LD_LIBRARY_PATH; command python  "$@"); }
python3() { (unset LD_LIBRARY_PATH; command python3 "$@"); }
pip()     { (unset LD_LIBRARY_PATH; command pip     "$@"); }
pip3()    { (unset LD_LIBRARY_PATH; command pip3    "$@"); }
```

2\. If `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126` results in cuda-related error for `jax`, then try installing the cpu-only pytorch: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu`