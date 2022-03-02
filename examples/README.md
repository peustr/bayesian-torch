## Running the examples

In this folder we give an example of applying a Bayesian extension to a deterministic CNN for image classification on MNIST. The deterministic CNN baseline is implemented in `cnn_mnist.py` and its Bayesian counterpart in `bcnn_mnist.py`. Notice that the two files only differ in the network definitions, and the inclusion of KL divergence in the loss function.

Run the examples using either of the following commands:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python cnn_mnist.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 python bcnn_mnist.py
```
The `CUBLAS_WORKSPACE_CONFIG=:4096:8` flag is added purely for reproducibility, notice that the two files contain the line `torch.use_deterministic_algorithms(True)`, which complements it. If you wish to run the experiments without the flag, make sure to comment out that line.

These two experiments should give the following accuracy scores on MNIST.
```
CNN:  Test set: Average loss: 0.0282, Accuracy: 9915/10000 (99%)
BCNN: Test set: Average loss: 0.0266, Accuracy: 9920/10000 (99%)
```
Evidently, the Bayesian neural network enjoys a tiny performance boost.
