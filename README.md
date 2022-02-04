# bayesian-torch

## Installation

If you have a pytorch environment, for example:
```bash
conda create -n myenv python=3.8
conda activate myenv
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

You may install this repository like so:
```bash
pip install git+https://github.com/peustr/bayesian-torch.git@main
```

## Usage

Replace the layers you want with the Bayesian ones by using the `bnn` module instead of `nn` like so:
```python
import torch.nn as nn
import btorch.bnn as bnn

model = nn.Sequential(
    bnn.Conv2d(inp_C, out_C, kernel_size),
    nn.ReLU(),
    bnn.Conv2d(out_C, 2 * out_C, kernel_size),
    nn.ReLU(),
    nn.Flatten(),
    bnn.Linear(num_inp_features, num_out_features),
)
```

Add the KL divergence loss term in your loss function, e.g.:
```python
import torch.nn.functional as F
from btorch.bnn.loss import kl_divergence

# ...
loss = F.cross_entropy(logits, target) + kl_divergence(model)
loss.backward()
```

Currently supported layers: `[bnn.Conv2d, bnn.Linear]`.
