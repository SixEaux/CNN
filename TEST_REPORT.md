# CNN correctness test report

Run on 2026-10-12 against the current local `main` branch. Application files in
`src/` were not modified, as requested.

## Result

**38 test methods: 28 passed, 10 did not pass.** Of the unsuccessful methods,
eight fail assertions and two raise exceptions. The unittest footer reports
`failures=10, errors=2` because the one-hot and training tests each contain two
failing subtests. These failures are deliberate checks of expected behavior,
not skipped tests or expected-failure annotations.

The dense and convolution numerical gradient checks pass for ordinary shapes,
including a complete CNN's gradients against derivatives of its mean loss.
Both configured model architectures complete forward/backward passes. All five
optimizers pass the update checks. Synthetic dataset preprocessing, save/load
prediction equivalence, dropout during inference, and a tiny CNN learning a
separable task also pass.

## Failures

| Test in `tests/test_cnn.py` | Expected behavior | Observed behavior | Source to inspect |
| --- | --- | --- | --- |
| `EvaluationTests.test_custom_validation_uses_its_own_sample_count` | Evaluate the supplied three validation images. | Uses the five-image test-set size; raises `IndexError` for index 3. This path is also called by training when validation is enabled. | `src/testing.py:45` |
| `EvaluationTests.test_evaluation_includes_final_batch_and_weights_metrics_by_sample_count` | With five images and batches of two, return five predictions, 80% accuracy and mean loss 3. | Returns four predictions; omits the last image. The fixture's omitted image is the only wrong prediction. | `src/testing.py:48` |
| `EvaluationTests.test_evaluation_accepts_batch_size_larger_than_dataset` | Five images with batch size eight form one smaller batch. | Processes no batches and raises `ValueError: need at least one array to concatenate`. | `src/testing.py:48` |
| `ModelAndTrainingTests.test_training_includes_final_partial_batch_and_small_datasets` | Train on all five images: batches of 3 and 2, or one batch of 5 when batch size is 8. | Batch size 3 processes only three images; batch size 8 performs no training updates. | `src/training.py:144` |
| `PoolingAndFlatteningTests.test_overlapping_pooling_accumulates_gradients` | Four windows selecting the same maximum contribute a total gradient of 4. | The maximum receives a gradient of 1. | `src/pooling.py:75` |
| `ConvolutionTests.test_backward_preserves_input_size_when_stride_leaves_unused_border` | A 6×6 input, kernel size 3, stride 2 and padding 1 receives a 6×6 input gradient. | Backward returns a 5×5 gradient. | `src/convolution.py:157` |
| `ConvolutionTests.test_rectangular_output_dimensions` | A 4×6 input with kernel size 3 and padding 1 records output dimensions `(4, 6, 2)`. | Records `(4, 4, 2)`, although forward produces `(4, 6, 2)`. | `src/convolution.py:68` |
| `PoolingAndFlatteningTests.test_rectangular_pooling_output_dimensions` | Pooling a 2×4 input with a 2×2 kernel records `(1, 2, 1)`. | Records `(1, 1, 1)`. | `src/pooling.py:29` |
| `ActivationAndLossTests.test_cross_entropy_remains_correct_for_extreme_wrong_logits` | Logits `[-1000, 1000]` with target class 0 have finite cross-entropy 2000. | Returns approximately 27.631 due to the probability floor inside the logarithm. | `src/loss.py:64` |
| `ActivationAndLossTests.test_one_hot_encoding_keeps_batch_axis_for_single_sample` | A single sample with three classes has one-hot shape `(1, 3)`. | Returns shape `(3,)` for both supported label layouts. The public helper loses its documented batch axis; current loss calculations can still work through broadcasting. | `src/loss.py:46` |

## Reproduce

From the repository root, the isolated environment prepared for this run works:

```bash
/tmp/cnn-tests-venv/bin/python -m unittest discover -s tests -v
```

To create it again:

```bash
python3 -m venv /tmp/cnn-tests-venv
/tmp/cnn-tests-venv/bin/python -m pip install -r requirements-test.txt
/tmp/cnn-tests-venv/bin/python -m unittest discover -s tests -v
```

The command currently exits with status 1 because the documented issues remain.
The temporary environment may be removed by the operating system. The existing
project `.venv` was left untouched: its metadata and packages target Python
3.12, but its interpreter symlink currently launches Python 3.14 and cannot
find those packages.

Verified environment: Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1, PyYAML 6.0.3,
tqdm 4.70.1 and Matplotlib 3.11.2. Tests use synthetic data, seeded parameters,
headless plotting imports and temporary save files. No MNIST download, full
dataset training, GUI interaction or saved-model overwrite was performed.
These checks cover core computations and workflows, not every possible input
or the model's accuracy on the real datasets.
