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

    # TODO: real datasets need to be updated
    if args.dataset in ('mnist', 'mnist_binary'):

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

        if args.dataset == 'mnist':
            output_dim = 9
        else:
            output_dim = 1

    elif args.dataset == 'iris_binary':

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
        output_dim = 1

    elif args.dataset == 'breastcancer_binary':

        bc = load_breast_cancer()
        X = bc.data
        y = bc.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        dataset_train = TensorDataset(Tensor(X_train), torch.as_tensor(y_train, dtype=torch.long))
        dataset_test = TensorDataset(Tensor(X_test), torch.as_tensor(y_test, dtype=torch.long))

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)

        input_dim = 30
        output_dim = 1

    elif args.dataset == 'logistic_synthetic':

        if args.dpower is None:
            if args.bias:
                args.input_dim = args.w_dim-1
            else:
                args.input_dim = args.w_dim
        else:
            args.input_dim = int(np.power(args.syntheticsamplesize, args.dpower))

        args.w_0 = torch.randn(args.input_dim, 1)

        if args.bias:
            args.b = torch.randn(1)
        else:
            args.b = torch.tensor([0.0])

        if args.posterior_viz:
            args.w_0 = torch.Tensor([[0.5], [1]])
            args.b = torch.tensor([0.0])

        args.output_dim = 1

        X = torch.randn(2*args.syntheticsamplesize, args.input_dim)
        output = torch.mm(X, args.w_0) + args.b
        m = torch.distributions.bernoulli.Bernoulli(torch.sigmoid(output))
        y = m.sample()

        # plt.plot(output.squeeze(dim=1).detach().numpy(), y.detach().numpy(), '.g')
        # plt.plot(output.squeeze(dim=1).detach().numpy(),softmax_output[:,0].detach().numpy(),'.r')
        # plt.title('synthetic logistic regression data: w^T x + b versus probabilities and Bernoulli(p)')
        # plt.show()

        #The splitting ratio of training set, validation set, testing set is 0.7, 0.15, 0.15
        train_size = args.syntheticsamplesize
        valid_size = int(args.syntheticsamplesize*0.5)
        test_size = 2*args.syntheticsamplesize - train_size - valid_size

        dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),[train_size, valid_size, test_size])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)

        args.loss_criterion = nn.BCELoss(reduction="sum")
        args.trueRLCT = (args.input_dim + 1*args.bias)/2

    elif args.dataset == 'tanh_synthetic':  # "Resolution of Singularities ... for Layered Neural Network" Aoyagi and Watanabe

        if args.dpower is None:
            args.H = int(args.w_dim/2)
        else:
            args.H = int(np.power(args.syntheticsamplesize, args.dpower)*0.5) #number of hidden unit
        args.a_params = torch.zeros([1, args.H], dtype=torch.float32)
        args.b_params = torch.zeros([args.H, 1], dtype=torch.float32)

        # what Watanabe calls three-layered neural network is actually one hidden layer
        # one input unit, H hidden units, and one output unit
        m = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        X = m.sample(torch.Size([2*args.syntheticsamplesize]))
        # w = {(a_m,b_m)}_{m=1}^p, p(y|x,w) = N(0,f(x,w)) where f(x,w) = \sum_{m=1}^p a_m tanh(b_m x)
        mean = torch.matmul(torch.tanh(torch.matmul(X, args.a_params)), args.b_params)
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
        args.input_dim = X.shape[1]
        args.output_dim = y.shape[1]

        args.loss_criterion = nn.MSELoss(reduction='sum')

        max_integer = int(math.sqrt(args.H))
        args.trueRLCT = (args.H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)

    elif args.dataset == 'reducedrank_synthetic':

        # suppose input_dimension=output_dimension + 3, H = output_dimension, H is number of hidden nuit
        # solve the equation (input_dimension + output_dimension)*H = np.power(args.syntheticsamplesize, args.dpower) to get output_dimension, then input_dimension, and H
        if args.dpower is None:
            args.output_dim = int((-3 + math.sqrt(9 + 4 * 2 * args.w_dim)) / 4)
        else:
            args.output_dim = int((-3 + math.sqrt(
                9 + 4 * 2 * np.power(args.syntheticsamplesize, args.dpower))) / 4)  # TODO: can easily be zero

        args.H = args.output_dim
        args.input_dim = args.output_dim + 3
        args.a_params = torch.transpose(
            torch.cat((torch.eye(args.H), torch.ones([args.H, args.input_dim - args.H], dtype=torch.float32)), 1), 0,
            1)  # input_dim * H
        args.b_params = torch.eye(args.output_dim)

        if args.w_dim == 2:
            args.a_params = torch.Tensor([1.0]).reshape(1, 1)
            args.b_params = torch.Tensor([1.0]).reshape(1, 1)
            args.input_dim = 1
            args.output_dim = 1
            args.H = 1
        # in this case, the rank r for args.b_params*args.a_params is H, output_dim + H < input_dim + r is satisfied

        m = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(args.input_dim)) #the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
        X = m.sample(torch.Size([2*args.syntheticsamplesize]))      
        mean = torch.matmul(torch.matmul(X, args.a_params), args.b_params)
        y_rv = MultivariateNormal(mean, torch.eye(args.output_dim))
        y = y_rv.sample()

        # The splitting ratio of training set, validation set, testing set is 0.7:0.15:0.15
        train_size = args.syntheticsamplesize
        valid_size = int(args.syntheticsamplesize*0.5)
        test_size = 2*args.syntheticsamplesize - train_size - valid_size

        dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),[train_size, valid_size, test_size])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)
        args.loss_criterion = nn.MSELoss(reduction='sum')
        args.trueRLCT = (args.output_dim * args.H - args.H ** 2 + args.input_dim * args.H) / 2 # rank r = H for the 'reducedrank_synthetic' dataset

    else:
        print('Not a valid dataset name. See options in dataset-factory')

    args.n = len(train_loader.dataset)

    return train_loader, valid_loader, test_loader



