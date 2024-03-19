import gc
import unittest
from unittest.mock import patch

from langchain_progress import MultiprocessingPBar
from .testing_utils import requires_tqdm


class TestMultiprocessingPbar(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    @requires_tqdm
    def test_init_with_tqdm(self):
        import tqdm

        total = 10
        tqdm_pbar = tqdm.tqdm(total=total)

        mp_pbar = MultiprocessingPBar(tqdm_pbar, total+2)
        self.assertEqual(mp_pbar.pbar, tqdm_pbar)

        # Check to see that total doesn't overwrite passed pbar
        self.assertEqual(mp_pbar.pbar.total, total)
        
    def test_init_with_non_tqdm_pbar(self):
        with self.assertRaises(TypeError) as e:
            MultiprocessingPBar('not a tqdm instance')

            self.assertEqual(str(e.exception), '`pbar` must be an instance of `tqdm.tqdm`.')

    def test_init_without_tqdm_installed(self):
        # Test with no existing tqdm but installed
        with patch('langchain_progress.multiprocessing_pbar.is_tqdm_installed', return_value=False):
            with self.assertRaises(ImportError) as e:
                MultiprocessingPBar()

                self.assertEqual(
                    str(e.exception), '`tqdm` is not installed. Please install `tqdm` to use '
                    'MultiprocessingPBar.')
                
    @requires_tqdm
    def test_enter(self):
        total = 10
        mp_pbar = MultiprocessingPBar(total=total)

        pbar = mp_pbar.__enter__()
        self.assertEqual(mp_pbar.context_alive.value, True)

        mp_pbar.__exit__(None, None, None)

    @requires_tqdm
    def test_exit(self):
        total = 10
        mp_bar = MultiprocessingPBar(total=total)

        self.assertFalse(mp_bar.pbar.disable)

        with mp_bar:
            pass

        self.assertEqual(mp_bar.context_alive.value, False)


if __name__ == '__main__':
    unittest.main()