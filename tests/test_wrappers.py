import gc
import unittest
from unittest.mock import patch

from langchain_progress.wrappers import PBarWrapper
from .testing_utils import (
    requires_tqdm,
    requires_ray,
    surpress_warning,
)


class TestPBarWrapper(unittest.TestCase):
    def setUp(self):
        self.texts = ['text1', 'text2', 'text3', 'text4', 'text5']

    def tearDown(self) -> None:
        gc.collect()

    def test_init_no_pbar_none_installed(self):
        '''Tests initialization with no progress bar provided and no packages installed.'''
        with (patch('langchain_progress.wrappers.is_tqdm_installed', return_value=False),
             patch('langchain_progress.wrappers.is_ray_installed', return_value=False)):
            with self.assertRaises(ImportError):
                PBarWrapper(self.texts, None)

    def test_init_no_pbar_no_tqdm_installed(self):
        '''Tests initialization with no progress bar provided and tqdm not installed.'''
        with patch('langchain_progress.wrappers.is_tqdm_installed', return_value=False):
            with self.assertRaises(ValueError):
                PBarWrapper(self.texts, None)

    @requires_tqdm
    @patch('logging.info')
    def test_init_tqdm_provided(self, mock_info):
        '''Tests initialization with a tqdm progress bar provided.'''
        from tqdm import tqdm

        pbar = tqdm(self.texts)
        wrapper = PBarWrapper(self.texts, pbar)

        self.assertIsInstance(wrapper.pbar, tqdm)
        self.assertFalse(mock_info.called)

        # Test update function
        wrapper._update(1)
        self.assertEqual(pbar.n, 1)


    # Surpress until https://github.com/ray-project/ray/issues/9546 is resolved
    @surpress_warning(warning=ResourceWarning)
    @requires_ray
    @patch('logging.info')
    def test_init_ray_provided(self, mock_info):
        '''Tests initialization with a ray progress bar provided.'''
        import ray
        from ray.experimental import tqdm_ray

        ray.init()
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        pbar = remote_tqdm.remote(total=len(self.texts))
        wrapper = PBarWrapper(self.texts, pbar)

        self.assertIsInstance(wrapper.pbar, ray.actor.ActorHandle)
        self.assertFalse(mock_info.called)

        # Test update function
        wrapper._update(1)
        tqdm_state = ray.get(pbar._get_state.remote())
        self.assertEqual(tqdm_state['x'], 1)
        ray.shutdown()

    @requires_tqdm
    @patch('logging.info')
    def test_init_no_pbar_provided(self, mock_info):
        '''Tests initialization with no tqdm progress bar provided.'''

        wrapper = PBarWrapper(self.texts, None)
        self.assertEqual(wrapper.pbar.total, len(self.texts))
        # mock_info.assert_called_once_with('No progress bar supplied. Creating a new tqdm progress bar.')

        # Test update function
        wrapper._update(1)
        self.assertEqual(wrapper.pbar.n, 1)

    @requires_tqdm
    def test_len(self):
        '''Tests the length of the wrapper.'''
        wrapper = PBarWrapper(self.texts, None)

        self.assertEqual(len(wrapper), len(self.texts))

    @requires_tqdm
    def test_iter(self):
        '''Tests iteration over the wrapper.'''
        wrapper = PBarWrapper(self.texts, None)

        iterator = iter(wrapper)
        self.assertIsInstance(iterator, PBarWrapper)

        for i in range(len(self.texts)):
            self.assertEqual(next(iterator), self.texts[i])

        with self.assertRaises(StopIteration):
            next(iterator)

    @requires_tqdm
    def test_getitem_slice(self):
        '''Tests getitem with slice index.'''
        wrapper = PBarWrapper(self.texts, None)

        sub_wrapper = wrapper[1:3]
        self.assertIsInstance(sub_wrapper, PBarWrapper)
        self.assertEqual(list(sub_wrapper), self.texts[1:3])

    @requires_tqdm
    def test_valid_getitem_int(self):
        '''Tests getitem with integer index.'''
        wrapper = PBarWrapper(self.texts, None)

        self.assertEqual(wrapper[0], self.texts[0])
        self.assertEqual(wrapper[-1], self.texts[-1])

    @requires_tqdm
    def test_out_of_bounds_getitem_int(self):
        '''Tests getitem with invalid integer index.'''
        wrapper = PBarWrapper(self.texts, None)

        with self.assertRaises(IndexError):
            wrapper[len(self.texts) + 1]


if __name__ == '__main__':
    unittest.main()