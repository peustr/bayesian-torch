## Running the examples

In this folder we give an example of applying a Bayesian extension to a deterministic CNN for image classification on MNIST.
The deterministic CNN baseline is implemented in `cnn_mnist.py` and its Bayesian counterpart in `bcnn_mnist.py`.
Notice that the two files only differ in the network definitions, and the inclusion of KL divergence in the loss function:
```bash
diff cnn_mnist.py bcnn_mnist.py

10a11,13
> import btorch.bnn as bnn
> from btorch.bnn.loss import kl_divergence
> from btorch.bnn.model_utils import create_zero_mean_unit_variance_prior
12c15,16
< class CNN(nn.Module):
---
>
> class BCNN(nn.Module):
15,18c19,22
<         self.conv1 = nn.Conv2d(1, 32, 3, 1)
<         self.conv2 = nn.Conv2d(32, 64, 3, 1)
<         self.fc1 = nn.Linear(12 * 12 * 64, 128)
<         self.fc2 = nn.Linear(128, 10)
---
>         self.conv1 = bnn.Conv2d(1, 32, 3, 1)
>         self.conv2 = bnn.Conv2d(32, 64, 3, 1)
>         self.fc1 = bnn.Linear(12 * 12 * 64, 128)
>         self.fc2 = bnn.Linear(128, 10)
30c34
< def train(args, model, device, train_loader, optimizer, epoch):
---
> def train(args, model, prior_model, device, train_loader, optimizer, epoch):
36c40,41
<         loss = F.cross_entropy(output, target)
---
>         nll = F.cross_entropy(output, target)
>         loss = nll + args.kld * kl_divergence(model, prior_model)
46c51
<                     loss.item(),
---
>                     nll.item(),
85a91
>     parser.add_argument("--kld", type=float, default=0.001, help="Discount factor for KL divergence.")
120c126,127
<         model = CNN().to(device)
---
>         model = BCNN().to(device)
>         prior_model = create_zero_mean_unit_variance_prior(model)
124c131
<             train(args, model, device, train_loader, optimizer, epoch)
---
>             train(args, model, prior_model, device, train_loader, optimizer, epoch)
131c138
<         torch.save(model.state_dict(), "cnn_mnist.pt")
---
>         torch.save(model.state_dict(), "bcnn_mnist.pt")
```

Run the examples using either of the following commands:
```bash
python cnn_mnist.py
python bcnn_mnist.py
```

These two experiments should give the following accuracy scores on MNIST.
```
CNN : Test accuracy: 99.14±0.05
BCNN: Test accuracy: 99.15±0.06
```
