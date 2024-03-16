import gc
import unittest

from langchain_progress import ProgressManager
from langchain_progress.utils import IMPLEMENTED_CLASSES


class MockPBarWrapper:
    def __init__(self, texts, pbar):
        return list(texts)


class TestProgressManager(unittest.TestCase):
    def setUp(self):
        self.texts = ['text1', 'text2', 'text3', 'text4', 'text5']

    def tearDown(self):
        gc.collect()

    def test_non_embedding_class(self):
        with self.assertRaises(TypeError):
            with ProgressManager(object()):
                pass

    # def test_embedding_classes(self):
    #     for cls in IMPLEMENTED_CLASSES:
    #         with self.subTest(cls=cls):
    #             self.run_test_for_class(cls)

    # def run_test_for_class(self, cls):
        
    #     with ProgressManager()
