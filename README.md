# CNN from scratch in Python

## Automated tests

From the repository root, create an isolated environment and run the suite:

```bash
python3 -m venv /tmp/cnn-tests-venv
/tmp/cnn-tests-venv/bin/python -m pip install -r requirements-test.txt
/tmp/cnn-tests-venv/bin/python -m unittest discover -s tests -v
```

The suite in `tests/test_cnn.py` checks layer outputs and numerical gradients,
losses, optimizers, data preprocessing, saving/loading, both configured CNNs,
and training/evaluation with small synthetic datasets. It does not download
MNIST, open GUI windows, or overwrite trained models. TensorFlow and pytest are
not required. The old `test/` folder contains ignored experiments rather than an
automated suite.

Failures identify behavior that needs fixing; application code is unchanged.
See [TEST_REPORT.md](TEST_REPORT.md) for the verified results and known failures.

## Datasets

This project uses the following datasets:

- **MNIST**: Yann LeCun et al.http://yann.lecun.com/exdb/mnist/
- **Fashion-MNIST**: Xiao et al.
  https://github.com/zalandoresearch/fashion-mnist
