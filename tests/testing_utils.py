from functools import wraps
from typing import Optional
import unittest
import warnings

from langchain_progress.utils import is_ray_installed, is_tqdm_installed

def requires_ray(test_case):
    '''Decorator to skip a test if ray is not installed'''
    return unittest.skipUnless(is_ray_installed(), 'Test requires ray')(test_case)
    

def requires_tqdm(test_case):
    '''Decorator to skip a test if tqdm is not installed'''
    return unittest.skipUnless(is_tqdm_installed(), 'Test requires tqdm')(test_case)


def surpress_warning(warning: Optional[Warning]=None):
    '''Decorator to suppress a warning in a test'''
    def decorator(test_case):
        @wraps(test_case)
        def wrapper(*args, **kwargs):
            warnings.simplefilter('ignore', category=warning if warning else Warning)
            return test_case(*args, **kwargs)
        return wrapper
    return decorator