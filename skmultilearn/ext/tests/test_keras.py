import unittest
import platform, sys

# no keras 32bit systems or python 2.7
if not (sys.version_info[0] == 2 or platform.architecture()[0] == "32bit"):
    from keras.models import Sequential
    from keras.layers import Dense
    from skmultilearn.ext.keras import Keras

    from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
    from skmultilearn.tests.classifier_basetest import ClassifierBaseTest

    KERAS_PARAMS = dict(epochs=1, batch_size=10, verbose=0)

    def create_model_multiclass(input_dim, output_dim):
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=input_dim, activation="relu"))
        model.add(Dense(output_dim, activation="softmax"))
        # Compile model
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def create_model_single_class(input_dim, output_dim):
        model = Sequential()
        model.add(Dense(12, input_dim=input_dim, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(output_dim, activation="sigmoid"))
        # Compile model
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    class KerasTest(ClassifierBaseTest):
        def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
            classifier = BinaryRelevance(
                classifier=Keras(create_model_single_class, False, KERAS_PARAMS),
                require_dense=[True, True],
            )

            self.assertClassifierWorksWithSparsity(classifier, "sparse")

        def test_if_dense_classification_works_on_non_dense_base_classifier(self):
            classifier = BinaryRelevance(
                classifier=Keras(create_model_single_class, False, KERAS_PARAMS),
                require_dense=[True, True],
            )

            self.assertClassifierWorksWithSparsity(classifier, "dense")

        def test_if_sparse_classification_works_on_dense_base_classifier(self):
            classifier = LabelPowerset(
                classifier=Keras(create_model_multiclass, True, KERAS_PARAMS),
                require_dense=[True, True],
            )

            self.assertClassifierWorksWithSparsity(classifier, "sparse")

        def test_if_dense_classification_works_on_dense_base_classifier(self):
            classifier = LabelPowerset(
                classifier=Keras(create_model_multiclass, True, KERAS_PARAMS),
                require_dense=[True, True],
            )

            self.assertClassifierWorksWithSparsity(classifier, "dense")

    if __name__ == "__main__":
        unittest.main()
