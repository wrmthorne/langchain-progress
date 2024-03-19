import gc
import unittest
from unittest.mock import patch

from langchain_progress.wrappers import (
    TRangeWrapper,
    PBarWrapper,
    Wrapper,
)
from .testing_utils import (
    requires_tqdm,
    requires_ray,
    surpress_warning,
)

class MockQueue:
    def __init__(self):
        self.n = 0

    def put(self, n):
        self.n += n


class TestWrapper(unittest.TestCase):
    def setUp(self):
        self.texts = ['text1', 'text2', 'text3', 'text4', 'text5']

    def tearDown(self) -> None:
        gc.collect()

    def test_init_no_pbar_none_installed(self):
        '''Tests initialization with no progress bar provided and no packages installed.'''
        with (patch('langchain_progress.wrappers.is_tqdm_installed', return_value=False),
              patch('langchain_progress.wrappers.is_ray_installed', return_value=False)):
            with self.assertRaises(ModuleNotFoundError):
                Wrapper(None)

    def test_init_no_pbar_no_tqdm_installed(self):
        '''Tests initialization with no progress bar provided and tqdm not installed.'''
        with patch('langchain_progress.wrappers.is_tqdm_installed', return_value=False):
            with self.assertRaises(ValueError):
                Wrapper(None)

    def test_init_multiprocessing_queue_as_pbar(self):
        '''Tests initialization with a multiprocessing queue as the progress bar.'''
        with self.assertLogs() as logs:
            wrapper = Wrapper(MockQueue(), total=len(self.texts))

        self.assertEqual(
            logs.output,
            ['INFO:langchain_progress.wrappers:Using multiprocessing progress bar. Assuming '
             'process is setup to consume updates.']
        )
        self.assertIsInstance(wrapper.pbar, MockQueue)
        self.assertEqual(wrapper._update.__name__, MockQueue().put.__name__)

    @requires_tqdm
    def test_init_tqdm_provided(self):
        '''Tests initialization with a tqdm progress bar provided.'''
        from tqdm import tqdm

        pbar = tqdm(self.texts)
        wrapper = Wrapper(pbar, total=len(self.texts))

        self.assertIsInstance(wrapper.pbar, tqdm)

        # Test update function
        wrapper._update(1)
        self.assertEqual(pbar.n, 1)

    # Surpress until https://github.com/ray-project/ray/issues/9546 is resolved
    @surpress_warning(warning=ResourceWarning)
    @requires_ray
    def test_init_ray_provided(self):
        '''Tests initialization with a ray progress bar provided.'''
        import ray
        from ray.experimental import tqdm_ray

        ray.init()
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        pbar = remote_tqdm.remote(total=len(self.texts))
        wrapper = Wrapper(pbar, total=len(self.texts))

        self.assertIsInstance(wrapper.pbar, ray.actor.ActorHandle)

        # Test update function
        wrapper._update(1)
        tqdm_state = ray.get(pbar._get_state.remote())
        self.assertEqual(tqdm_state['x'], 1)
        ray.shutdown()

    @requires_tqdm
    def test_init_no_pbar_provided(self):
        '''Tests initialization with no tqdm progress bar provided.'''

        with self.assertLogs() as logs:
            wrapper = Wrapper(None, total=len(self.texts))

        self.assertEqual(wrapper.pbar.total, len(self.texts))
        self.assertEqual(
            logs.output,
            ['INFO:langchain_progress.wrappers:No progress bar supplied. Creating a new tqdm progress bar.']
        )

        # Test update function
        wrapper._update(1)
        self.assertEqual(wrapper.pbar.n, 1)


class TestPBarWrapper(unittest.TestCase):
    def setUp(self):
        self.texts = ['text1', 'text2', 'text3', 'text4', 'text5']

    def tearDown(self) -> None:
        gc.collect()

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


class TestTRangeWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_queue = MockQueue()
        self.wrapper = TRangeWrapper(self.mock_queue)

    def tearDown(self):
        gc.collect()

    def test_call_batch_size_one(self):
        '''Tests the call method of the wrapper.'''
        wrapper = self.wrapper(5)

        self.assertEqual(wrapper.index, -1)
        self.assertEqual(wrapper.stop, 5)

        for i in wrapper:
            self.assertEqual(i, wrapper.index)

        self.assertEqual(i, 4)

    def test_call_with_final_batch_overflow(self):
        '''Tests the call method of the wrapper with a final batch that overflows.'''
        step_size = 2
        wrapper = self.wrapper(0, 5, step_size)

        self.assertEqual(wrapper.index, -step_size)
        self.assertEqual(wrapper.stop, 5)

        for i in wrapper:
            self.assertEqual(i, wrapper.index)

        # Checks to see that the number of pushes to the pbar matches the stop value
        self.assertEqual(self.mock_queue.n, 5)



if __name__ == '__main__':
    unittest.main()