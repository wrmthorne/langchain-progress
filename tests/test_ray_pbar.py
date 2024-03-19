import gc
import unittest
from unittest.mock import patch

from langchain_progress import RayPBar
from .testing_utils import requires_ray


class TestRayPBar(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_init_without_tqdm_installed(self):
        # Test with no existing tqdm but installed
        with patch('langchain_progress.ray_pbar.is_ray_installed', return_value=False):
            with self.assertRaises(ImportError) as e:
                RayPBar()

                self.assertEqual(
                    str(e.exception), '`ray` is not installed. Please install `ray` to use RayPBar.')
    
    @requires_ray
    def test_init_with_ray_installed(self):
        import ray

        r_pbar = RayPBar(total=10)

        self.assertEqual(r_pbar.total, 10)
        ray.shutdown()

    @requires_ray
    def test_enter_exit(self):
        import ray
        
        r_pbar = RayPBar(total=10)

        pbar = r_pbar.__enter__()
        self.assertIsInstance(r_pbar.pbar, ray.actor.ActorHandle)

        pbar.update.remote(1)
        tqdm_state = ray.get(pbar._get_state.remote())
        self.assertEqual(tqdm_state['x'], 1)

        r_pbar.__exit__(None, None, None)
        ray.shutdown()