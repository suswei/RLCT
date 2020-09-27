# wiseodd/natural-gradients

import torch.nn.functional as F

import numpy as np
from numpy.linalg import inv

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from matplotlib import pyplot as plt


class args():

    def __init__(self):
        self.n = 100
        self.batchsize = 10
        self.epochs = 1000

        self.output_dim = 1
        self.input_dim = 2

        self.H = 5

        self.H0 = 3
        self.y_std = 0.1

        self.numbetas = 5

        self.R = 5

args = args()

a = Normal(0.0, 1.0)
args.a_params = 0.2 * a.sample((args.input_dim, args.H0))
b = Normal(0.0, 1.0)
args.b_params = 0.2 * b.sample((args.H0,args.output_dim))
m = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(
    args.input_dim))  # the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
X = 3.0 * m.sample(torch.Size([2 * args.n]))

mean = torch.matmul(torch.matmul(X, args.a_params), args.b_params)
y_rv = MultivariateNormal(torch.zeros(args.output_dim), torch.eye(args.output_dim))
y = mean + args.y_std * y_rv.sample(torch.Size([2 * args.n]))

# The splitting ratio of training set, validation set, testing set is 0.7:0.15:0.15
train_size = args.n
valid_size = int(args.n * 0.5)
test_size = 2 * args.n - train_size - valid_size

dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),
                                                                           [train_size, valid_size, test_size])

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True)

args.loss_criterion = nn.MSELoss(reduction='sum')
args.w_dim = (args.input_dim+args.output_dim+args.H)*args.H

wholex = train_loader.dataset[:][0]
wholey = train_loader.dataset[:][1]


class Model(nn.Module):

    def __init__(self, input_dim, output_dim, H):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(input_dim, H, bias=False)
        self.fc2 = nn.Linear(H, H, bias=False)
        self.fc3 = nn.Linear(H, output_dim, bias=False)

        self.W = [self.fc1.weight, self.fc2.weight, self.fc3.weight]

    def forward(self, x):
        a1 = self.fc1(x)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        z = self.fc3(h2)

        cache = (a1, h1, a2, h2)

        z.retain_grad()
        for c in cache:
            c.retain_grad()

        return z, cache

nll = np.empty((args.numbetas, args.R))
temperature = np.empty((args.numbetas))
alphas = np.linspace(0.1, 0.5, args.numbetas)
temperature = (alphas / 2) * (args.n / args.batchsize - 1)
print('temperature {}'.format(temperature))

for alpha_index in range(0, args.numbetas):

    alpha = alphas[alpha_index]

    for r in range(0, args.R):

        model = Model(args.input_dim, args.output_dim, args.H)

        print('Training {}/{} ensemble at {}/{} temp'.format(r + 1, args.R, alpha_index + 1, args.numbetas))

        A = []  # KFAC A
        G = []  # KFAC G

        A_inv, G_inv = 3*[0], 3*[0]

        for Wi in model.W:
            A.append(torch.zeros(Wi.size(1)))
            G.append(torch.zeros(Wi.size(0)))

        eps = 10
        inverse_update_freq = 20

        # Training
        for epoch in range(1, args.epochs):

            for batch_idx, (data, target) in enumerate(train_loader):

                # Forward
                z, cache = model.forward(data)
                a1, h1, a2, h2 = cache

                # Loss
                criterion = nn.MSELoss(reduction='sum')
                loss = criterion(z, target)

                # wd = torch.tensor(0.)
                # for p in model.parameters():
                #     # anchor = anchor_dist.sample(p.shape)
                #     # wd += ((p - anchor) ** 2).sum()
                #     wd += (p ** 2).sum()
                # loss = closs + wd
                loss.backward()
                # print(f'Iter-{epoch}; Loss: {loss:.3f}')

                # KFAC matrices
                G1_ = 1/args.batchsize * a1.grad.t() @ a1.grad
                A1_ = 1/args.batchsize * data.t() @ data
                G2_ = 1/args.batchsize * a2.grad.t() @ a2.grad
                A2_ = 1/args.batchsize * h1.t() @ h1
                G3_ = 1/args.batchsize * z.grad.t() @ z.grad
                A3_ = 1/args.batchsize * h2.t() @ h2

                G_ = [G1_, G2_, G3_]
                A_ = [A1_, A2_, A3_]

                # Update running estimates of KFAC
                rho = min(1-1/epoch, 0.95)

                for k in range(3):
                    A[k] = rho*A[k] + (1-rho)*A_[k]
                    G[k] = rho*G[k] + (1-rho)*G_[k]

                # Step
                for k in range(3):
                    # Amortize the inverse. Only update inverses every now and then
                    if (epoch-1) % inverse_update_freq == 0:
                        A_inv[k] = (A[k] + eps*torch.eye(A[k].shape[0])).inverse()
                        G_inv[k] = (G[k] + eps*torch.eye(G[k].shape[0])).inverse()

                    delta = G_inv[k] @ model.W[k].grad.data @ A_inv[k]
                    model.W[k].data -= alpha * delta

                # PyTorch stuffs
                model.zero_grad()

            if epoch % 100 == 0:
                output, _ = model.forward(wholex)
                print('Epoch {}: training mse {}'.format(epoch, args.loss_criterion(wholey, output)/args.n))

        # Record nll
        final_output, _ = model.forward(wholex)
        nll[alpha_index, r] = ((wholey - final_output) ** 2).sum() / (2 * (args.y_std ** 2))

design_x = np.vstack((np.ones(args.numbetas), temperature)).T
design_y = np.mean(nll,1)
design_y = design_y[:, np.newaxis]
fit = inv(design_x.T.dot(design_x)).dot(design_x.T).dot(design_y)
ols_intercept_estimate = fit[0][0]
RLCT_estimate =fit[1][0]

print('RLCT estimate: {}'.format(RLCT_estimate))

plt.scatter(temperature, np.mean(nll, 1), label='nll beta')
plt.plot(temperature, ols_intercept_estimate + RLCT_estimate * temperature, 'b-', label='ols')
plt.title("d_on_2 = {} , hat lambda ols = {:.1f}".format(args.w_dim / 2, RLCT_estimate), fontsize=8)
plt.xlabel("1/beta", fontsize=8)
plt.ylabel("E^beta_w [nL_n(w)]", fontsize=8)
plt.legend()
plt.show()