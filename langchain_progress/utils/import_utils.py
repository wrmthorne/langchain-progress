import importlib
from typing import Union

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from tqdm import tqdm

# Type alias for the progress bar
ProgressBar = Union['ActorHandle', 'tqdm']

def is_ray_installed() -> bool:
    '''Determines if ray is installed'''
    return importlib.util.find_spec('ray') is not None

def is_tqdm_installed() -> bool:
    '''Determines if tqdm is installed'''
    return importlib.util.find_spec('tqdm') is not None