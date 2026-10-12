"""Correctness tests for the CNN, using tiny synthetic data and no downloads.

Run from the repository root: python -m unittest discover -s tests -v
Failing tests deliberately expose bugs; they are not marked as expected failures.
"""

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.activation import Activation
from src.convolution import Convolutional
from src.dense import Dense
from src.dropout import Dropout
from src.flattening import Flattening
from src.helpers import get_layer, get_optimizer, load_config
from src.import_data import Dataset, import_data, load_from_npz
from src.loss import Loss
from src.model import Model
from src.optimizer import ADAM, Adagrad, Optimizer, RMSprop, SGD, SGD_momentum
from src.pooling import MaxPool
from src.save_load_model import load_model, save_model
from src.testing import Testing as Evaluator
from src.training import Training


def _numerical_gradient(function, values, step=1e-5):
    """Differentiate a scalar objective without using the layer's backward pass."""
    gradient = np.zeros_like(values, dtype=np.float64)
    for index in np.ndindex(values.shape):
        original = values[index]
        try:
            values[index] = original + step
            plus = function()
            values[index] = original - step
            minus = function()
            gradient[index] = (plus - minus) / (2 * step)
        finally:
            values[index] = original
    return gradient


def _reference_convolution(images, weight, bias, stride, padding):
    """Slow, explicit NHWC cross-correlation oracle, independent of tensordot."""
    padded = np.pad(images, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    kernel_height, kernel_width, _, kernels = weight.shape
    height = (padded.shape[1] - kernel_height) // stride + 1
    width = (padded.shape[2] - kernel_width) // stride + 1
    result = np.empty((images.shape[0], height, width, kernels))
    for batch in range(images.shape[0]):
        for row in range(height):
            for column in range(width):
                window = padded[
                    batch,
                    row * stride : row * stride + kernel_height,
                    column * stride : column * stride + kernel_width,
                    :,
                ]
                for kernel in range(kernels):
                    result[batch, row, column, kernel] = (
                        np.sum(window * weight[:, :, :, kernel]) + bias[0, 0, 0, kernel]
                    )
    return result


def _dataset(train_size=5, test_size=5, validation_size=3):
    def samples(size):
        images = np.arange(size, dtype=np.float64).reshape(size, 1, 1, 1)
        labels = (np.arange(size) % 2).reshape(size, 1)
        return images, labels

    train_images, train_values = samples(train_size)
    test_images, test_values = samples(test_size)
    validation_images, validation_values = samples(validation_size)
    return Dataset(
        train_images, train_values, validation_images, validation_values,
        test_images, test_values, {0: "zero", 1: "one"},
    )


class _PredictableModel:
    """Evaluation oracle: predicts parity, with sample i having loss i + 1."""

    def __init__(self):
        self.layers = []
        self.seen = []
        self.batch_sizes = []

    def forward(self, images, expected=None, test=False):
        indices = images[:, 0, 0, 0].astype(int)
        if not test:
            self.seen.extend(indices.tolist())
        logits = np.eye(2)[indices % 2]
        return logits, (indices + 1).astype(float).reshape(-1, 1)

    def choice(self, logits):
        return np.argmax(logits, axis=1, keepdims=True)

    def backward(self, batch_size, batch_images=None):
        self.batch_sizes.append(batch_size)


class _CNNTestCase(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)
        self.enterContext(patch("src.model.print_network"))
        for module in ("src.testing", "src.training"):
            self.enterContext(patch(f"{module}.trange", side_effect=lambda n, **kwargs: range(n)))

    def make_model(self, layers, data=None, classes=2):
        model = Model(layers, Loss("CEL", classes), "synthetic", data or _dataset())
        # Override random initializers so the tests give reproducible results.
        for layer in layers:
            if hasattr(layer, "weight"):
                layer.weight = self.rng.normal(0, 0.1, layer.weight.shape)
                layer.bias = np.zeros_like(layer.bias, dtype=np.float64)
        return model


class DenseTests(_CNNTestCase):
    def test_forward_matches_hand_calculated_values(self):
        layer = Dense(2)
        layer.initial_param(3, 2)
        layer.weight = np.array([[1., 2., 3.], [-1., 0., 1.]])
        layer.bias = np.array([[0.5, -0.5]])
        assert_allclose(layer.forward(np.array([[1., 2., 3.], [0., 1., 0.]])),
                        [[14.5, 1.5], [2.5, -0.5]])

    def test_backward_matches_numerical_input_weight_and_bias_gradients(self):
        layer = Dense(2)
        layer.initial_param(3, 2)
        layer.weight = self.rng.normal(size=(2, 3))
        layer.bias = self.rng.normal(size=(1, 2))
        images = self.rng.normal(size=(2, 3))
        upstream = self.rng.normal(size=(2, 2))
        layer.forward(images)
        input_gradient = layer.backward(upstream, batch_size=2)
        objective = lambda: np.sum(layer.forward(images) * upstream)
        assert_allclose(input_gradient, _numerical_gradient(objective, images), atol=1e-7)
        assert_allclose(layer.dW, _numerical_gradient(objective, layer.weight) / 2, atol=1e-7)
        assert_allclose(layer.dB, _numerical_gradient(objective, layer.bias) / 2, atol=1e-7)

    def test_gradient_computation_does_not_update_parameters(self):
        layer = Dense(2)
        layer.initial_param(3, 2)
        original_weight, original_bias = layer.weight.copy(), layer.bias.copy()
        layer.forward(np.ones((2, 3)))
        weight_gradient, bias_gradient = layer.backward(np.ones((2, 2)), 2, just_computation=True)
        assert_allclose(weight_gradient, np.ones((2, 3)))
        assert_allclose(bias_gradient, np.ones((1, 2)))
        assert_array_equal(layer.weight, original_weight)
        assert_array_equal(layer.bias, original_bias)


class ConvolutionTests(_CNNTestCase):
    def make_layer(self, shape, kernel, stride, padding):
        layer = Convolutional(2, kernel, stride=stride, padding=padding)
        layer.initial_param(shape[1:], (1, 1, 2))
        layer.weight = self.rng.normal(size=(kernel, kernel, shape[-1], 2))
        layer.bias = self.rng.normal(size=(1, 1, 1, 2))
        return layer, self.rng.normal(size=shape)

    def test_forward_matches_loop_oracle_with_channels_stride_and_padding(self):
        for shape, kernel, stride, padding in (
            ((2, 4, 4, 2), 3, 1, 1),
            ((2, 5, 5, 2), 3, 2, 0),
            ((2, 4, 6, 2), 2, 2, 1),
        ):
            with self.subTest(shape=shape, kernel=kernel, stride=stride, padding=padding):
                layer, images = self.make_layer(shape, kernel, stride, padding)
                expected = _reference_convolution(images, layer.weight, layer.bias, stride, padding)
                assert_allclose(layer.forward(images), expected, atol=1e-10)

    def check_gradients(self, shape, kernel, stride, padding):
        layer, images = self.make_layer(shape, kernel, stride, padding)
        output = layer.forward(images)
        upstream = self.rng.normal(size=output.shape)
        input_gradient = layer.backward(upstream, batch_size=shape[0])
        self.assertEqual(input_gradient.shape, images.shape)
        objective = lambda: np.sum(
            _reference_convolution(images, layer.weight, layer.bias, stride, padding) * upstream
        )
        assert_allclose(input_gradient, _numerical_gradient(objective, images), atol=1e-7)
        assert_allclose(layer.dW, _numerical_gradient(objective, layer.weight) / shape[0], atol=1e-7)
        assert_allclose(layer.dB, _numerical_gradient(objective, layer.bias) / shape[0], atol=1e-7)

    def test_backward_matches_numerical_gradients(self):
        for shape, kernel, stride, padding in (
            ((2, 4, 4, 2), 3, 1, 1), ((2, 5, 5, 2), 3, 2, 0),
        ):
            with self.subTest(stride=stride, padding=padding):
                self.check_gradients(shape, kernel, stride, padding)

    def test_backward_preserves_input_size_when_stride_leaves_unused_border(self):
        self.check_gradients((1, 6, 6, 1), 3, 2, 1)

    def test_one_by_one_kernel_backward_matches_numerical_gradients(self):
        self.check_gradients((1, 3, 3, 1), 1, 1, 0)

    def test_rectangular_output_dimensions(self):
        layer, images = self.make_layer((1, 4, 6, 1), 3, 1, 1)
        self.assertEqual(layer.out_dim, layer.forward(images).shape[1:])


class PoolingAndFlatteningTests(_CNNTestCase):
    def test_pooling_forward_and_backward_route_gradients_to_maxima(self):
        layer = MaxPool(2)
        layer.initial_param((4, 4, 1))
        images = np.arange(1, 17, dtype=float).reshape(1, 4, 4, 1)
        assert_array_equal(layer.forward(images).reshape(2, 2), [[6, 8], [14, 16]])
        expected = np.zeros_like(images)
        expected[0, 1::2, 1::2, 0] = [[1, 2], [3, 4]]
        assert_array_equal(layer.backward(np.arange(1, 5).reshape(1, 2, 2, 1)), expected)

    def test_overlapping_pooling_accumulates_gradients(self):
        layer = MaxPool(2, stride=1)
        layer.initial_param((3, 3, 1))
        images = np.array([[1., 2., 3.], [4., 9., 5.], [6., 7., 8.]]).reshape(1, 3, 3, 1)
        output = layer.forward(images)
        expected = np.zeros_like(images)
        expected[0, 1, 1, 0] = 4
        assert_array_equal(layer.backward(np.ones_like(output)), expected)

    def test_rectangular_pooling_output_dimensions(self):
        layer = MaxPool(2)
        layer.initial_param((2, 4, 1))
        self.assertEqual(layer.out_dim, (1, 2, 1))
        assert_array_equal(layer.forward(np.arange(8.).reshape(1, 2, 4, 1)),
                           np.array([5., 7.]).reshape(1, 1, 2, 1))

    def test_flattening_preserves_values_and_restores_gradient_shape(self):
        layer = Flattening()
        layer.initial_param((2, 3, 2))
        images = np.arange(24.).reshape(2, 2, 3, 2)
        self.assertEqual(layer.out_dim, 12)
        assert_array_equal(layer.forward(images), np.arange(24.).reshape(2, 12))
        assert_array_equal(layer.backward(np.arange(24.).reshape(2, 12)), images)


class ActivationAndLossTests(_CNNTestCase):
    def test_activation_outputs_and_numerical_gradients(self):
        inputs = np.array([[-2., -0.5, 0.25, 2.]])  # Avoid the ReLU kink at zero.
        upstream = np.array([[0.3, -0.2, 0.4, 0.7]])
        references = {
            "relu": np.maximum(inputs, 0),
            "tanh": np.tanh(inputs),
            "sigmoid": 1 / (1 + np.exp(-inputs)),
        }
        for name, expected in references.items():
            with self.subTest(activation=name):
                layer = Activation(name)
                assert_allclose(layer.forward(inputs), expected)
                gradient = layer.backward(upstream)
                objective = lambda: np.sum(layer.forward(inputs) * upstream)
                assert_allclose(gradient, _numerical_gradient(objective, inputs), atol=1e-7)

    def test_invalid_activation_and_loss_are_rejected(self):
        with self.assertRaises(ValueError):
            Activation("unknown").forward(np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            Loss("unknown", 2).forward(np.zeros((1, 2)), np.array([[0]]))

    def test_softmax_is_normalized_and_invariant_to_large_constant_shift(self):
        loss = Loss("CEL", 3)
        logits = np.array([[1., 2., 3.], [-2., 0., 2.]])
        probabilities = loss.softmax(logits)
        assert_allclose(probabilities.sum(axis=1), np.ones(2))
        assert_allclose(loss.softmax(logits + 10000), probabilities)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_loss_values_and_gradients(self):
        logits = np.array([[0.2, -0.3, 0.7], [0.1, 0.9, -0.4]])
        labels = np.array([[2], [1]])
        for name in ("CEL", "MSEL"):
            with self.subTest(loss=name):
                loss = Loss(name, 3)
                values = loss.forward(logits, labels)
                if name == "CEL":
                    expected = np.log(np.exp(logits).sum(axis=1)) - logits[np.arange(2), labels[:, 0]]
                else:
                    expected = np.sum((logits - np.eye(3)[labels[:, 0]]) ** 2, axis=1) / 2
                assert_allclose(values[:, 0], expected)
                gradient = loss.backward()
                objective = lambda: np.sum(loss.forward(logits, labels))
                assert_allclose(gradient, _numerical_gradient(objective, logits), atol=1e-7)

    def test_cross_entropy_remains_correct_for_extreme_wrong_logits(self):
        # log(exp(-1000) + exp(1000)) - (-1000) = 2000.
        value = Loss("CEL", 2).forward(np.array([[-1000., 1000.]]), np.array([[0]]))
        self.assertTrue(np.isfinite(value).all())
        assert_allclose(value, [[2000.]])

    def test_one_hot_encoding_keeps_batch_axis_for_single_sample(self):
        loss = Loss("CEL", 3)
        for labels in (np.array([1]), np.array([[1]])):
            with self.subTest(shape=labels.shape):
                assert_array_equal(loss.one_hot_vector(labels), [[0., 1., 0.]])

    def test_dropout_uses_same_scaled_mask_in_forward_and_backward(self):
        layer = Dropout(0.5)
        layer.generator = np.random.default_rng(42)
        output = layer.forward(np.ones((4, 8)))
        self.assertEqual(set(np.unique(output)), {0., 2.})
        assert_array_equal(layer.backward(np.full((4, 8), 3.)), output * 3)
        assert_array_equal(Dropout(0).forward(np.ones((2, 3))), np.ones((2, 3)))


class OptimizerTests(_CNNTestCase):
    def test_two_updates_match_hand_calculated_values_for_all_optimizers(self):
        cases = (
            (SGD(0.1), [0.9, -0.8], 0.45),
            (SGD_momentum(0.1, momentum_rate=0.5), [0.8, -0.6], 0.4),
            (Adagrad(0.1), [0.9 + 0.1 / np.sqrt(5), -0.9 - 0.2 / np.sqrt(20)], 0.4 + 0.05 / np.sqrt(1.25)),
            (RMSprop(0.1, beta=0.5), [0.9 + 0.1 / np.sqrt(2), -0.9 - 0.2 / np.sqrt(8)], 0.4 + 0.05 / np.sqrt(0.5)),
            (ADAM(0.1, momentum_rate=0.5, velocity_rate=0.5), [0.9, -0.9], 0.4),
        )
        for optimizer, expected_weight, expected_bias in cases:
            with self.subTest(optimizer=type(optimizer).__name__):
                layer = Dense(1)
                layer.weight, layer.bias = np.array([[1., -1.]]), np.array([[0.5]])
                layer.dW, layer.dB = np.array([[2., -4.]]), np.array([[1.]])
                optimizer.update(layer, 0.1)
                layer.dW, layer.dB = np.array([[-1., 2.]]), np.array([[-0.5]])
                optimizer.update(layer, 0.1)
                assert_allclose(layer.weight, [expected_weight], atol=1e-8)
                assert_allclose(layer.bias, [[expected_bias]], atol=1e-8)

    def test_optimizer_state_is_independent_between_layers(self):
        optimizer = ADAM(0.1)
        layers = [Dense(1), Dense(1)]
        for layer in layers:
            layer.weight, layer.bias = np.ones((1, 1)), np.ones((1, 1))
            layer.dW, layer.dB = np.ones((1, 1)), np.ones((1, 1))
        optimizer.update(layers[0], 0.1)
        optimizer.update(layers[0], 0.1)
        optimizer.update(layers[1], 0.1)
        assert_allclose(layers[0].weight, [[0.8]])
        assert_allclose(layers[1].weight, [[0.9]])

    def test_optimizers_skip_layers_without_parameters(self):
        for optimizer in (SGD(0.1), SGD_momentum(0.1), Adagrad(0.1), RMSprop(0.1), ADAM(0.1)):
            with self.subTest(optimizer=type(optimizer).__name__):
                optimizer.update(Activation("relu"), 0.1)

    def test_learning_rate_schedules(self):
        cases = (("exponential", 0.1 * np.exp(-0.6)), ("inverse", 0.1 / 1.6), ("step", 0.02))
        for method, expected in cases:
            with self.subTest(method=method):
                optimizer = Optimizer(0.1, decay_rate=0.2, decay_steps=2, lr_decay_method=method)
                self.assertAlmostEqual(optimizer.get_learning_rate(3), expected)
        self.assertEqual(SGD(0.1).get_learning_rate(100), 0.1)


class DataAndPersistenceTests(_CNNTestCase):
    def setUp(self):
        super().setUp()
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.train_images = np.arange(24, dtype=np.uint8).reshape(6, 2, 2, 1)
        self.train_values = np.arange(6).reshape(6, 1)
        np.savez(self.folder / "mnist.npz", train_images=self.train_images,
                 train_values=self.train_values, test_images=self.train_images[:2],
                 test_values=self.train_values[:2])

    def test_normalization_uses_training_statistics(self):
        for method in ("", "division", "center-reduction"):
            with self.subTest(method=method):
                data = load_from_npz(self.folder / "mnist.npz", method)
                if method == "division":
                    expected = self.train_images / 255
                elif method == "center-reduction":
                    expected = (self.train_images - self.train_images.mean()) / self.train_images.std()
                else:
                    expected = self.train_images
                assert_allclose(data.train_images, expected)
                assert_allclose(data.test_images, expected[:2])

    def test_split_preserves_image_label_pairs_without_overlap(self):
        with patch("numpy.random.permutation", return_value=np.array([5, 2, 0, 4, 1, 3])):
            data = import_data("mnist", validation_part=1 / 3, data_root=self.folder)
        self.assertEqual(len(data.train_images), 4)
        self.assertEqual(len(data.validation_images), 2)
        self.assertEqual(data.train_images.dtype, np.float32)
        train_ids, validation_ids = data.train_values[:, 0], data.validation_values[:, 0]
        self.assertTrue(set(train_ids).isdisjoint(validation_ids))
        self.assertEqual(set(train_ids) | set(validation_ids), set(range(6)))
        for images, labels in ((data.train_images, train_ids), (data.validation_images, validation_ids)):
            assert_allclose(images * 255, self.train_images[labels], atol=1e-6)

    def test_invalid_dataset_normalization_and_validation_fraction_are_rejected(self):
        with self.assertRaises(ValueError):
            import_data("unknown", data_root=self.folder)
        with self.assertRaises(FileNotFoundError):
            import_data("fashion_mnist", data_root=self.folder)
        with self.assertRaises(ValueError):
            import_data("mnist", data_root=self.folder, normalization_method="unknown")
        for fraction in (-0.1, 1):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                import_data("mnist", data_root=self.folder, validation_part=fraction)

    def test_save_load_round_trip_preserves_predictions_and_configuration(self):
        config = {"model": {"layers": [{"type": "Flattening"}, {"type": "Dense", "number_neurons": 2}]}}
        model = self.make_model([get_layer(item) for item in config["model"]["layers"]])
        images = self.rng.normal(size=(3, 1, 1, 1))
        expected, _ = model.forward(images, test=True)
        with patch("src.save_load_model.FOLDER_PATH", str(self.folder)):
            save_model(model.layers, copy.deepcopy(config), "round_trip")
            parameters, saved_config = load_model("round_trip")
            with parameters:
                restored = self.make_model([get_layer(item) for item in saved_config["model"]["layers"]])
                for index, layer in enumerate(restored.layers):
                    if hasattr(layer, "weight"):
                        layer.weight = parameters[f"layer_{index}_weight"].copy()
                        layer.bias = parameters[f"layer_{index}_bias"].copy()
            actual, _ = restored.forward(images, test=True)
        assert_array_equal(actual, expected)
        self.assertEqual(saved_config["model"], config["model"])
        self.assertTrue(saved_config["initialized"])
        self.assertEqual(saved_config["save_name"], "round_trip")


class ModelAndTrainingTests(_CNNTestCase):
    def test_configured_mnist_and_fashion_models_can_forward_and_backward(self):
        for filename in ("config_mnist.yaml", "config_fashion_mnist.yaml"):
            with self.subTest(config=filename):
                config = load_config(filename)
                original = copy.deepcopy(config)
                layers = [get_layer(item) for item in config["model"]["layers"]]
                data = _dataset()._replace(train_images=np.zeros((2, 28, 28, 1)))
                model = self.make_model(layers, data, classes=config["nb_classes"])
                for layer in layers:
                    if isinstance(layer, Dropout):
                        layer.generator = np.random.default_rng(42)
                logits, losses = model.forward(self.rng.normal(size=(2, 28, 28, 1)), np.array([[0], [1]]))
                self.assertEqual(logits.shape, (2, config["nb_classes"]))
                self.assertEqual(losses.shape, (2, 1))
                model.backward(2)
                for layer in layers:
                    if hasattr(layer, "weight"):
                        self.assertEqual(layer.dW.shape, layer.weight.shape)
                        self.assertEqual(layer.dB.shape, layer.bias.shape)
                        self.assertTrue(np.isfinite(layer.dW).all())
                        self.assertTrue(np.isfinite(layer.dB).all())
                get_optimizer(config["training"]["optimizer"])
                self.assertEqual(config, original)

    def test_inference_skips_dropout(self):
        dropout = Dropout(0.5)
        model = self.make_model([Flattening(), dropout, Dense(2)])
        images = np.ones((2, 1, 1, 1))
        with patch.object(dropout, "forward", side_effect=AssertionError("Dropout used in inference")):
            first, loss = model.forward(images, test=True)
            second, _ = model.forward(images, test=True)
        self.assertIsNone(loss)
        assert_array_equal(first, second)

    def test_model_choice_returns_one_label_per_image(self):
        model = self.make_model([Flattening(), Dense(2)])
        assert_array_equal(model.choice(np.array([[3., 1.], [-2., 4.]])), [[0], [1]])

    def test_complete_cnn_parameter_gradients_match_mean_loss_derivatives(self):
        images = self.rng.normal(size=(2, 4, 4, 1))
        values = np.array([[0], [1]])
        data = _dataset()._replace(train_images=images)
        model = self.make_model(
            [Convolutional(2, 2), Activation("tanh"), MaxPool(2), Flattening(), Dense(2)], data,
        )
        model.forward(images, values)
        model.backward(2)
        objective = lambda: np.mean(model.forward(images, values)[1])
        for index, layer in enumerate(model.layers):
            if hasattr(layer, "weight"):
                with self.subTest(layer=index, parameter="weight"):
                    assert_allclose(layer.dW, _numerical_gradient(objective, layer.weight), atol=1e-7)
                with self.subTest(layer=index, parameter="bias"):
                    assert_allclose(layer.dB, _numerical_gradient(objective, layer.bias), atol=1e-7)

    def test_tiny_cnn_learns_a_separable_task(self):
        images = np.concatenate((-np.ones((2, 4, 4, 1)), np.ones((2, 4, 4, 1))))
        values = np.array([[0], [0], [1], [1]])
        data = _dataset()._replace(train_images=images, train_values=values,
                                  test_images=images.copy(), test_values=values.copy())
        model = self.make_model([Convolutional(2, 3, padding=1), Activation("tanh"), Flattening(), Dense(2)], data)
        evaluator = Evaluator("synthetic", model, data)
        initial_loss = evaluator.exam()[1]
        trainer = Training("synthetic", model, evaluator, SGD(0.2), data)
        # The training shuffle cannot change the order-independent full-batch update.
        with patch("numpy.random.permutation", side_effect=lambda n: np.arange(n)):
            trainer.train(epoch=30, batch_size=4)
        accuracy, loss, predictions = evaluator.exam()
        self.assertLess(loss, initial_loss / 5)
        self.assertEqual(accuracy, 100)
        assert_array_equal(predictions, values)
        self.assertEqual(trainer.finished_epochs, 30)

    def test_early_stopping_requires_validation_data(self):
        data, model = _dataset(), _PredictableModel()
        with self.assertRaises(ValueError):
            Training("synthetic", model, Evaluator("synthetic", model, data), SGD(0.1), data, early_stop=True)

    def test_training_includes_final_partial_batch_and_small_datasets(self):
        for batch_size in (3, 8):
            with self.subTest(batch_size=batch_size):
                data, model = _dataset(), _PredictableModel()
                trainer = Training("synthetic", model, Evaluator("synthetic", model, data), SGD(0.1), data)
                with patch("numpy.random.permutation", side_effect=lambda n: np.arange(n)):
                    trainer.train(epoch=1, batch_size=batch_size)
                self.assertEqual(sorted(model.seen), list(range(5)))
                self.assertEqual(model.batch_sizes, [3, 2] if batch_size == 3 else [5])


class EvaluationTests(_CNNTestCase):
    def setUp(self):
        super().setUp()
        self.data = _dataset()
        # Only the final prediction is wrong: losing it silently changes 80% to 100%.
        values = self.data.test_values.copy()
        values[-1, 0] = 1 - values[-1, 0]
        self.data = self.data._replace(test_values=values)
        self.evaluator = Evaluator("synthetic", _PredictableModel(), self.data)

    def test_default_evaluation_uses_all_samples(self):
        accuracy, loss, predictions = self.evaluator.exam()
        self.assertEqual(accuracy, 80)
        self.assertEqual(loss, 3)
        self.assertEqual(predictions.shape, (5, 1))

    def test_evaluation_includes_final_batch_and_weights_metrics_by_sample_count(self):
        accuracy, loss, predictions = self.evaluator.exam(batch_size=2)
        self.assertEqual(predictions.shape, (5, 1))
        self.assertEqual(accuracy, 80)
        self.assertEqual(loss, 3)

    def test_evaluation_accepts_batch_size_larger_than_dataset(self):
        accuracy, loss, predictions = self.evaluator.exam(batch_size=8)
        self.assertEqual(predictions.shape, (5, 1))
        self.assertEqual(accuracy, 80)
        self.assertEqual(loss, 3)

    def test_custom_validation_uses_its_own_sample_count(self):
        accuracy, loss, predictions = self.evaluator.exam(
            self.data.validation_images, self.data.validation_values,
        )
        self.assertEqual(predictions.shape, (3, 1))
        self.assertEqual(accuracy, 100)
        self.assertEqual(loss, 2)


if __name__ == "__main__":
    unittest.main()
