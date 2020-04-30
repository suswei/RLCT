from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, SubsetRandomSampler
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt

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

        X = torch.randn(2*args.syntheticsamplesize, input_dim)
        output = torch.mm(X, args.w_0) + args.b
        output_cat_zero = torch.cat((output, torch.zeros(X.shape[0], 1)), 1)
        softmax_output = F.softmax(output_cat_zero, dim=1)
        m = torch.distributions.bernoulli.Bernoulli(softmax_output[:,0])
        y = m.sample()
        y = y.type(torch.LongTensor) # otherwise torch nll_loss complains

        # plt.plot(output.squeeze(dim=1).detach().numpy(), y.detach().numpy(), '.g')
        # plt.plot(output.squeeze(dim=1).detach().numpy(),softmax_output[:,0].detach().numpy(),'.r')
        # plt.title('synthetic logistic regression data: w^T x + b versus probabilities and Bernoulli(p)')
        # plt.show()

        #The splitting ratio of training set, validation set, testing set is 0.7:0.15:0.15
        train_size = args.syntheticsamplesize
        valid_size = int(args.syntheticsamplesize*0.5)
        test_size = 2*args.syntheticsamplesize - train_size - valid_size

        dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),[train_size, valid_size, test_size])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)

        loss_criterion = nn.NLLLoss(reduction="mean")
        trueRLCT = (input_dim + 1)/2

    elif args.dataset == 'tanh_synthetic':  # "Resolution of Singularities ... for Layered Neural Network" Aoyagi and Watanabe

        # what Watanabe calls three-layered neural network is actually one hidden layer
        # one input unit, H hidden units, and one output unit
        m = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        X = m.sample(torch.Size([2*args.syntheticsamplesize]))

        # w = {(a_m,b_m)}_{m=1}^p, p(y|x,w) = N(0,f(x,w)) where f(x,w) = \sum_{m=1}^p a_m tanh(b_m x)
        mean = torch.matmul(torch.tanh(torch.matmul(X, torch.transpose(args.a_params,0,1))), torch.transpose(args.b_params,0,1))
        y_rv = Normal(mean,1)

        y = y_rv.sample()

        # The splitting ratio of training set, validation set, testing set is 0.7:0.15:0.15
        train_size = args.syntheticsamplesize
        valid_size = int(args.syntheticsamplesize*0.5)
        test_size = 2*args.syntheticsamplesize - train_size - valid_size

        dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y), [train_size, valid_size, test_size])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        loss_criterion = nn.MSELoss(reduction='mean')

        max_integer = int(math.sqrt(args.H))
        trueRLCT = (args.H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)

    elif args.dataset == 'reducedrank_synthetic':
        m = MultivariateNormal(torch.zeros(args.H + 3), torch.eye(args.H + 3)) #the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
        X = m.sample(torch.Size([2*args.syntheticsamplesize]))      
        mean = torch.matmul(torch.tanh(torch.matmul(X, torch.transpose(args.a_params,0,1))), torch.transpose(args.b_params,0,1))

        y_rv = MultivariateNormal(mean, torch.eye(args.H)) #output_dim equals H

        y = y_rv.sample()

        # The splitting ratio of training set, validation set, testing set is 0.7:0.15:0.15
        train_size = args.syntheticsamplesize
        valid_size = int(args.syntheticsamplesize*0.5)
        test_size = 2*args.syntheticsamplesize - train_size - valid_size

        dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),[train_size, valid_size, test_size])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        loss_criterion = nn.MSELoss(reduction='mean')
        trueRLCT = (output_dim * args.H - args.H ** 2 + input_dim * args.H) / 2 # rank r = H for the 'reducedrank_synthetic' dataset

    else:
        print('Not a valid dataset name. See options in dataset-factory')

    return train_loader, valid_loader, test_loader, input_dim, output_dim, loss_criterion, trueRLCT



