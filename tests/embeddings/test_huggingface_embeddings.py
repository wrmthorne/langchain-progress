import os
import unittest

from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceHubEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceInstructEmbeddings,
)
import responses

from tests.embeddings.utils import EmbeddingsTester, EmbeddingsTesterMixin
from tests.testing_utils import requires_instructor_embedding


RESPONSES_PATH = os.path.join(os.path.dirname(__file__), 'responses.yaml')


class TestHuggingFaceBgeEmbeddings(EmbeddingsTesterMixin, unittest.TestCase):
    def setUp(self):
        self.tester = EmbeddingsTester(self)

    def test_with_encode(self):
        self.tester.test_embedding_class(
            embedding_class=HuggingFaceBgeEmbeddings,
            wrapped_class_name='client',
            wrapped_method_name='encode',
        )


class TestHuggingFaceEmbeddings(EmbeddingsTesterMixin, unittest.TestCase):
    def setUp(self):
        self.tester = EmbeddingsTester(self)

    def test_with_encode(self):
        self.tester.test_embedding_class(
            embedding_class=HuggingFaceEmbeddings,
            wrapped_class_name='client',
            wrapped_method_name='encode',
        )

    def test_with_encode_multi_process(self):
        self.tester.test_embedding_class(
            embedding_class=HuggingFaceEmbeddings,
            wrapped_class_name='client',
            wrapped_method_name='encode_multi_process',
            multi_process=True,
        )


class TestHuggingFaceHubEmbeddings(EmbeddingsTesterMixin, unittest.TestCase):
    def setUp(self):
        self.tester = EmbeddingsTester(self)

    @responses.activate
    def test_with_post(self):
        responses._add_from_file(file_path=RESPONSES_PATH)

        self.tester.test_unwrappable_embedding_class(
            embedding_class=HuggingFaceHubEmbeddings,
            wrapped_class_name='client',
            wrapped_method_name='post',
        )

    @responses.activate
    def test_logging(self):
        responses._add_from_file(file_path=RESPONSES_PATH)

        self.tester.test_embedding_class_logging(
            embedding_class=HuggingFaceHubEmbeddings,
            levels=['WARNING'],
            messages=['wrapped texts cannot be sent to remote endpoint. Continuing without progress bar.'],
        )


class TestHuggingFaceInferenceAPIEmbeddings(EmbeddingsTesterMixin, unittest.TestCase):
    def setUp(self):
        self.tester = EmbeddingsTester(self)

    @responses.activate
    def test_logging(self):
        responses._add_from_file(file_path=RESPONSES_PATH)

        self.tester.test_embedding_class_logging(
            embedding_class=HuggingFaceInferenceAPIEmbeddings,
            levels=['WARNING'],
            messages=['wrapped texts cannot be sent to remote endpoint. Continuing without progress bar.'],
            api_key='fake_key',
        )


class TestHuggingFaceInstructEmbeddings(EmbeddingsTesterMixin, unittest.TestCase):
    def setUp(self):
        self.tester = EmbeddingsTester(self)

    @requires_instructor_embedding
    def test_with_encode(self):
        self.tester.test_embedding_class(
            embedding_class=HuggingFaceInstructEmbeddings,
            wrapped_class_name='client',
            wrapped_method_name='encode',
        )


if __name__ == '__main__':
    unittest.main()