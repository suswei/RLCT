from __future__ import print_function
import torch
from torchvision import datasets, transforms
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, SubsetRandomSampler
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

def get_dataset_by_id(args,kwargs):

    if args.dataset in ('MNIST', 'MNIST-binary'):

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batchsize, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batchsize, shuffle=True, **kwargs)

        # to know the dataset better
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        input_dim = images.shape[2]*images.shape[3]

        if args.dataset == 'MNIST':
            output_dim = 10  # TODO: how do I extract this from the dataloader?
        else:
            output_dim = 2

    elif args.dataset == 'iris-binary':

        iris = load_iris()
        X = iris.data
        y = iris.target
        y[y == 2] = 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        dataset_train = TensorDataset(Tensor(X_train), torch.as_tensor(y_train, dtype=torch.long))
        dataset_test = TensorDataset(Tensor(X_test), torch.as_tensor(y_test, dtype=torch.long))

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)

        input_dim = 4
        output_dim = 2

    elif args.dataset == 'breastcancer-binary':

        bc = load_breast_cancer()
        X = bc.data
        y = bc.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        dataset_train = TensorDataset(Tensor(X_train), torch.as_tensor(y_train, dtype=torch.long))
        dataset_test = TensorDataset(Tensor(X_test), torch.as_tensor(y_test, dtype=torch.long))

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)

        input_dim = 30
        output_dim = 2

    elif args.dataset == 'lr_synthetic':

        output_dim = 2
        input_dim = args.w_0.shape[0]

        X = torch.randn(args.syntheticsamplesize, input_dim)
        output = torch.mm(X, args.w_0) + args.b
        output_cat_zero = torch.cat((output, torch.zeros(X.shape[0], 1)), 1)
        softmax_output = F.softmax(output_cat_zero, dim=1)
        y = softmax_output.data.max(1)[1]  # get the index of the max probability

        train_size = int(0.8 * args.syntheticsamplesize)
        test_size = args.syntheticsamplesize - train_size

        dataset_train, dataset_test = torch.utils.data.random_split(TensorDataset(X, y), [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        #
        # dataset_train = TensorDataset(Tensor(X_train), torch.as_tensor(y_train, dtype=torch.long))
        # dataset_test = TensorDataset(Tensor(X_test), torch.as_tensor(y_test, dtype=torch.long))
        #
        # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        # test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)

    elif args.dataset == '3layertanh_synthetic':  # "Resolution of Singularities ... for Layered Neural Network" Aoyagi and Watanabe

        # three-layered neural network of one input unit, p hidden units, and one output unit
        m = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        X = m.sample(args.syntheticsamplesize)

        p = 20
        a_params = torch.randn(args.syntheticsamplesize, p)
        b_params = torch.randn(args.syntheticsamplesize, p)

        # w = {(a_m,b_m)}_{m=1}^p, p(y|x,w) = N(0,f(x,w)) where f(x,w) = \sum_{m=1}^p a_m tanh(b_m x)

        mean = a_params*F.tanh(b_params * X.repeat(1,p))
        y_rv = Normal(mean,1)

        y = y_rv.sample()

        train_size = int(0.8 * args.syntheticsamplesize)
        test_size = args.syntheticsamplesize - train_size

        dataset_train, dataset_test = torch.utils.data.random_split(TensorDataset(X, y), [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)

    else:
        print('Not a valid dataset name. See options in dataset-factory')



    return train_loader, test_loader, input_dim, output_dim

