""" Based on https://github.com/pytorch/examples/blob/main/mnist/main.py. """

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import btorch.bnn as bnn
from btorch.bnn.loss import kl_divergence


class BCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = bnn.Conv2d(1, 32, 3, 1)
        self.conv2 = bnn.Conv2d(32, 64, 3, 1)
        self.fc1 = bnn.Linear(12 * 12 * 64, 128)
        self.fc2 = bnn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='sum') + kl_divergence(model, reduction='sum')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example.')
    parser.add_argument(
        '--batch-size', type=int, default=64, help='Input batch size for training (default: 64).')
    parser.add_argument(
        '--test-batch-size', type=int, default=64, help='Input batch size for testing (default: 64).')
    parser.add_argument(
        '--epochs', type=int, default=5, help='Number of epochs to train (default: 5).')
    parser.add_argument(
        '--lr', type=float, default=1.0, help='Learning rate (default: 1.0).')
    parser.add_argument(
        '--gamma', type=float, default=0.5, help='Learning rate scheduler step gamma (default: 0.5).')
    parser.add_argument(
        '--seed', type=int, default=1, help='Random seed (default: 1).')
    parser.add_argument(
        '--log-interval', type=int, default=10, help='How many batches to wait before logging training status.')
    parser.add_argument(
        '--save-model', action='store_true', default=False, help='Save the final model.')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = BCNN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "bcnn_mnist.pt")


if __name__ == '__main__':
    main()
