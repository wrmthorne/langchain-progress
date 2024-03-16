import gc
import unittest

from langchain_progress import ProgressManager
from langchain_progress.progress_manager import _AttributeState

    
class MockEmbeddingClass:
    def __init__(self):
        self.original_value = 'original_value'

    def original_method(self):
        return 'original_method'


class TestAttributeState(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_mutate_and_restore_value(self):
        cls = MockEmbeddingClass()
        attr_name = 'original_value'
        mutated_value = 'mutated_value'

        state = _AttributeState(cls, attr_name, mutated_value)
        self.assertEqual(cls.original_value, 'mutated_value')

        state.restore_state()
        self.assertEqual(cls.original_value, 'original_value')

    def test_mutate_and_restore_method(self):
        cls = MockEmbeddingClass()
        attr_name = 'original_method'
        mutated_value = lambda: 'mutated_method'

        state = _AttributeState(cls, attr_name, mutated_value)
        self.assertEqual(cls.original_method(), 'mutated_method')

        state.restore_state()
        self.assertEqual(cls.original_method(), 'original_method')


class TestProgressManager(unittest.TestCase):
    def test_non_embedding_class(self):
        with self.assertRaises(TypeError):
            with ProgressManager(object()):
                pass



if __name__ == '__main__':
    unittest.main()