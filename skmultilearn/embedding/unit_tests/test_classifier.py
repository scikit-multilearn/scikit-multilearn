import unittest

from skmultilearn.adapt import MLkNN
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.embedding import CLEMS, SKLearnEmbedder, OpenNetworkEmbedder, EmbeddingClassifier
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest
from sklearn.linear_model import LinearRegression
from sklearn.manifold import SpectralEmbedding
from copy import copy

class EmbeddingTest(ClassifierBaseTest):
    TEST_NEIGHBORS = 3

    def classifiers(self):
        graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)

        for embedding in OpenNetworkEmbedder._EMBEDDINGS:
            yield EmbeddingClassifier(
                    OpenNetworkEmbedder(copy(graph_builder), embedding, 4, 'add', True),
                    LinearRegression(),
                    MLkNN(k=2)
                )

        yield EmbeddingClassifier(
            SKLearnEmbedder(SpectralEmbedding(n_components=2)),
            LinearRegression(),
            MLkNN(k=2)
        )

        yield EmbeddingClassifier(
            CLEMS('cost', 2),
            LinearRegression(),
            MLkNN(k=2),
            True
        )

    def test_if_embedding_classification_works_on_sparse_input(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithSparsity(classifier, 'sparse')
            self.assertClassifierPredictsProbabilities(classifier, 'sparse')

    def test_if_embedding_classification_works_on_dense_input(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithSparsity(classifier, 'dense')
            self.assertClassifierPredictsProbabilities(classifier, 'dense')


    def test_if_embedding_works_with_cross_validation(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
