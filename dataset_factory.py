from __future__ import print_function
import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

def get_dataset_by_id(args,kwargs):

    if args.dataset_name in ('MNIST', 'MNIST-binary'):

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # to know the dataset better
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        input_dim = images.shape[2]*images.shape[3]

        if args.dataset_name == 'MNIST':
            output_dim = 10  # TODO: how do I extract this from the dataloader?
        else:
            output_dim = 2

    elif args.dataset_name == 'iris-binary':

        iris = load_iris()
        X = iris.data
        y = iris.target
        y[y == 2] = 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        dataset_train = TensorDataset(Tensor(X_train), torch.as_tensor(y_train, dtype=torch.long))
        dataset_test = TensorDataset(Tensor(X_test), torch.as_tensor(y_test, dtype=torch.long))

        # dataset_train = TensorDataset(Tensor(X_train), Tensor(y_train,dtype=torch.long))
        # dataset_test = TensorDataset(Tensor(X_test), Tensor(y_test,dtype=torch.long))

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, **kwargs)

        input_dim = 4
        output_dim = 2

    else:
        print('Not a valid dataset name. See options in dataset-factory')



    return train_loader, test_loader, input_dim, output_dim

