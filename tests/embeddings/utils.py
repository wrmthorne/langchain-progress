import gc
from unittest.mock import patch

from langchain_progress import ProgressManager


class MockPBarWrapper(list):
    def __init__(self, texts, pbar):
        super().__init__(texts)


class EmbeddingsTesterMixin:
    @classmethod
    def setUpClass(cls):
        cls.texts = ['text1', 'text2', 'text3', 'text4', 'text5']

    def tearDown(self):
        gc.collect()


class EmbeddingsTester:
    def __init__(self, parent):
        self.parent = parent

    @patch('langchain_progress.progress_manager.PBarWrapper', MockPBarWrapper)
    def test_embedding_class(self, embedding_class=None, wrapped_class_name=None, wrapped_method_name=None, *model_args, **model_kwargs):
        embedding_instance = embedding_class(*model_args, **model_kwargs)

        wrapped_class = getattr(embedding_instance, wrapped_class_name, embedding_instance)

        self.parent.assertNotEqual(getattr(wrapped_class, wrapped_method_name).__func__, ProgressManager._wrapped_method)

        with ProgressManager(embedding_instance):
            self.parent.assertEqual(getattr(wrapped_class, wrapped_method_name).__func__, ProgressManager._wrapped_method)
            vectors = embedding_instance.embed_documents(self.parent.texts)

        self.parent.assertEqual(len(vectors), len(self.parent.texts))
        self.parent.assertNotEqual(getattr(wrapped_class, wrapped_method_name).__func__, ProgressManager._wrapped_method)


    @patch('langchain_progress.progress_manager.PBarWrapper', MockPBarWrapper)
    def test_unwrappable_embedding_class(self, embedding_class=None, wrapped_class_name=None, wrapped_method_name=None, *model_args, **model_kwargs):
        embedding_instance = embedding_class(*model_args, **model_kwargs)

        wrapped_class = getattr(embedding_instance, wrapped_class_name, embedding_instance)

        self.parent.assertNotEqual(getattr(wrapped_class, wrapped_method_name).__func__, ProgressManager._wrapped_method)

        with ProgressManager(embedding_instance):
            self.parent.assertNotEqual(getattr(wrapped_class, wrapped_method_name).__func__, ProgressManager._wrapped_method)
            vectors = embedding_instance.embed_documents(self.parent.texts)

        self.parent.assertEqual(len(vectors), len(self.parent.texts))
        self.parent.assertNotEqual(getattr(wrapped_class, wrapped_method_name).__func__, ProgressManager._wrapped_method)


    @patch('langchain_progress.progress_manager.PBarWrapper', MockPBarWrapper)
    def test_embedding_class_logging(self, embedding_class=None, levels=['WARNING'], messages=[''], *model_args, **model_kwargs):
        embedding_instance = embedding_class(*model_args, **model_kwargs)

        with self.parent.assertLogs() as logs:
            with ProgressManager(embedding_instance):
                embedding_instance.embed_documents(self.parent.texts)

            self.parent.assertEqual(len(logs.output), len(levels))
            for level, message, log in zip(levels, messages, logs.output):
                self.parent.assertIn(level, log)
                self.parent.assertIn(message, log)