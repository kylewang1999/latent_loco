import os, time, io, base64
import hydra
import logging
import rich.logging as rlogging
import imageio
import numpy as np
from IPython.display import HTML
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.sdk.wandb_run import Run
from pathlib import Path
import mujoco
from brax.io import mjcf

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax, jax.numpy as jnp
from functools import wraps
import flax.nnx as nnx
import orbax.checkpoint as ocp
from typing import Callable


def _warp_angle(th: jnp.ndarray, center: float = jnp.pi) -> jnp.ndarray:
    return jnp.mod(th + center, 2 * jnp.pi) - center


def get_repo_root():
    return os.path.dirname((os.path.abspath(__file__)))


def load_xml_as_mjmodel(xml_or_path: str|Path, 
                        mj_moodel_kwargs: dict = {},
                        *, return_brax_sys: bool = False):
    """
    Load a MuJoCo model from a filesystem path or an XML string.

    - If `xml_or_path` points to an existing file, use `from_xml_path` (resolves <include>).
    - Otherwise, treat it as an XML string and use `from_xml_string` (no include resolution).
    
    Inputs:
        xml_or_path: str | Path
        mj_moodel_kwargs: dict: optional kwargs for the MuJoCo model'
        return_brax_sys: bool

    Returns:
        mj_model                          if return_brax_sys=False
        (mj_model, brax_sys)              if return_brax_sys=True
    """
    assert mj_moodel_kwargs.get('solver', 'mjSOL_CG') in set([
        'mjSOL_CG', 'mjSOL_NEWTON', 'mjSOL_PGS'
    ])
    
    s = str(xml_or_path)
    p = Path(s)

    if p.exists():
        mj_model = mujoco.MjModel.from_xml_path(str(p))
    else:
        if "<mujoco" not in s:
            raise FileNotFoundError(
                f"{xml_or_path!r} is neither an existing file nor a valid MuJoCo XML string."
            )
        mj_model = mujoco.MjModel.from_xml_string(s)

    mj_model.opt.solver = getattr(mujoco.mjtSolver, mj_moodel_kwargs.get('solver', 'mjSOL_CG'))
    mj_model.opt.iterations = mj_moodel_kwargs.get('solver_iter', 6)
    mj_model.opt.ls_iterations = mj_moodel_kwargs.get('solver_ls_iter', 6)

    if return_brax_sys:
        sys = mjcf.load_model(mj_model)  # convert to Brax System
        return mj_model, sys
    return mj_model


def make_log_dir(description=''):
    os.makedirs(log_dir:=f'{get_repo_root()}/logs/{description}_{time.strftime("%m%d_%H%M")}', 
                exist_ok=True)
    return log_dir


def get_console_logger(name="rssm_torch", level="INFO"):
    ''' Get a colored logger for the console 
    Input:
        name: Name of the logger. Note that if name == "" then the logger will be the root logger
            - Set name="" to see logs produced by other libraries
        level: Logging level (NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL)
    Example::
        logger = get_console_logger('super/data.py')
        logger.debug("Debug message")
        logger.info("Information message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    for handler in logger.handlers:
        logger.removeHandler(handler)

    handler = rlogging.RichHandler(level=level, markup=True)
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent jax, orbax, flax, genesis from logging to this logger
    logging.getLogger('jax').propagate = False
    logging.getLogger('orbax').propagate = False
    logging.getLogger('flax').propagate = False
    logging.getLogger('genesis').propagate = False
    return logger


def load_configs(config_dir=os.path.join(get_repo_root(), 'configs'),
                 config_name='base') -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name=config_name)
    return cfg


def save_configs(cfg, save_path='./configs.yaml'):
    with open(save_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    CONSOLE_LOGGER.info(f"Configs saved to {save_path}")


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        CONSOLE_LOGGER.info(f"{func.__name__} took {execution_time:.2f} seconds to execute")
        return result
    return wrapper


def catch_keyboard_interrupt(message: str = "Training interrupted by user"):
    """
    Decorator to handle KeyboardInterrupt gracefully in training functions.
    
    Args:
        message: Custom message to print when interrupted
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                print(message)
                return None
        return wrapper
    return decorator


def display_video(imgs_array, save=False, save_path='./video.mp4'):
    
    imgs_array = imgs_array.astype(np.uint8)
    if imgs_array.shape[-1] == 1:
        imgs_array = np.repeat(imgs_array, 3, axis=-1)
    
    # Ensure imgs_array is a list/sequence of images
    if isinstance(imgs_array, np.ndarray):
        # If it's a single numpy array, convert to list
        if len(imgs_array.shape) == 3:  # Single image (H, W, C)
            imgs_array = [imgs_array]
        elif len(imgs_array.shape) == 4:  # Multiple images (N, H, W, C)
            imgs_array = list(imgs_array)
        else:
            raise ValueError(f"Unexpected image array shape: {imgs_array.shape}")

    
    # save to memory buffer
    buf = io.BytesIO()
    imageio.mimsave(buf, imgs_array, fps=30, format='mp4')
    buf.seek(0)
    
    if save:
        assert save_path.endswith('.mp4'), "save_path must end with .mp4"
        imageio.mimsave(save_path, imgs_array, fps=30, format='mp4')
        CONSOLE_LOGGER.info(f"Video saved to {save_path}")

    data = base64.b64encode(buf.getvalue()).decode('ascii')
    html = f'<video src="data:video/mp4;base64,{data}" controls></video>'
    return HTML(html)


def wandb_log_artifact(run:Run, type, path):
    artifact = wandb.Artifact(f"{run.name}_{type}", type=type)
    artifact.add_file(path)
    run.log_artifact(artifact)

''' nnx module saving/loading '''

def save_nnx_module(model: nnx.Module, save_dir: str):
    
    save_dir = ocp.test_utils.erase_and_create_empty(save_dir)
    _, state, _ = nnx.split(model, nnx.Not(nnx.RngState), nnx.RngState)
    nnx.display(state)
    
    ckpt_dir = ocp.test_utils.erase_and_create_empty(save_dir)
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(save_dir / "state", args=ocp.args.StandardSave(state))
    checkpointer.wait_until_finished()
    CONSOLE_LOGGER.info(f"NNX module saved to {save_dir}")


def restore_nnx_module(build_fn: Callable[[], nnx.Module],
                       ckpt_dir: str | Path,
                       *,
                       name: str = "state") -> nnx.Module:
    
    path = Path(ckpt_dir) / name
    checkpointer = ocp.StandardCheckpointer()

    abstract_model = nnx.eval_shape(build_fn)
    graphdef, state_norng, state_rng = nnx.split(abstract_model, nnx.Not(nnx.RngState), nnx.RngState)
    nnx.display(state_norng)

    state_restored = checkpointer.restore(path, state_norng, strict=False)
    model = nnx.merge(graphdef, state_restored, state_rng)
    return model

CONSOLE_LOGGER = get_console_logger()


if __name__ == '__main__':
    pass