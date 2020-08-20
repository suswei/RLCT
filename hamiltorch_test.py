import torch
import hamiltorch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from numpy.linalg import inv

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

hamiltorch.set_random_seed(123)
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class Net(nn.Module):
#
#     def __init__(self, layer_sizes, loss='multi_class', bias=True):
#         super(Net, self).__init__()
#         self.layer_sizes = layer_sizes
#         self.layer_list = []
#         self.loss = loss
#         self.bias = bias
#         self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1], bias=self.bias)
#
#     #         self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2],bias = self.bias)
#
#     def forward(self, x):
#         x = self.l1(x)
#         #         x = torch.softmax(x)
#         #         x = self.l2(x)
#         return x
#
#
# #         output = self.layer_list[-1](x)
# #         if self.loss is 'binary_class' or 'regression':
# #             return output
# #         if self.loss is 'multi_class':
# #         return F.log_softmax(x, dim=1)
# layer_sizes = [4, 3]
# net = Net(layer_sizes)


class ToyResLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()
        aprime = torch.Tensor(1)
        bprime = torch.Tensor(1)
        self.aprime = nn.Parameter(aprime)
        self.bprime = nn.Parameter(bprime)
        # self.size_in, self.size_out = size_in, size_out
        # weights = torch.Tensor(size_out, size_in)
        # self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        # bias = torch.Tensor(size_out)
        # self.bias = nn.Parameter(bias)

        # initialize aprime, bprime
        nn.init.uniform_(self.aprime)
        nn.init.uniform_(self.bprime)
        # nn.init.kaiming_uniform_(self.weights) # weight init
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        # bound = 1 / np.sqrt(fan_in)
        # nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w = (self.aprime ** 3) * (self.aprime - 3*self.bprime + 27*(self.bprime ** 3))
        return x*w


class ToyRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.ToyResLayer = ToyResLayer()

    def forward(self, x):
        return self.ToyResLayer(x)


class Tanh(nn.Module):

    def __init__(self, input_dim, output_dim, H):

        super(Tanh, self).__init__()
        self.fc1 = nn.Linear(input_dim, H, bias=False)
        self.fc2 = nn.Linear(H, output_dim, bias=False)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


net = Tanh(1,1,1)
# net = ToyRes()
print(net)
trueRLCT = 0.5
w_dim = 2

# from sklearn.datasets import load_iris
# import numpy as np
#
# np.random.seed(0)
# data = load_iris()
# x_ = data['data']
# y_ = data['target']
# N_tr = 10  # 50
# N_val = 140
# a = np.arange(x_.shape[0])
# train_index = np.random.choice(a, size=N_tr, replace=False)
# val_index = np.delete(a, train_index, axis=0)
# x_train = x_[train_index]
# y_train = y_[train_index]
# x_val = x_[val_index][:]
# y_val = y_[val_index][:]
# x_m = x_train.mean(0)
# x_s = x_train.std(0)
# x_train = (x_train - x_m) / x_s
# x_val = (x_val - x_m) / x_s
# D_in = x_train.shape[1]
# x_train = torch.FloatTensor(x_train)
# y_train = torch.FloatTensor(y_train)
# x_val = torch.FloatTensor(x_val)
# y_val = torch.FloatTensor(y_val)
# plt.scatter(x_train.numpy()[:, 0], y_train.numpy())
#
# x_train = x_train.to(device)
# y_train = y_train.to(device)
# x_val = x_val.to(device)
# y_val = y_val.to(device)

# dataset
n = 100
m = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
x_train = m.sample(torch.Size([n]))
x_val = m.sample(torch.Size([n]))

# generate target from N(0,1)
y_rv = Normal(0.0, 0.1)  # torch.distributions.normal.Normal(loc, scale) where scale is standard deviation
y_train = 0.01 * y_rv.sample(torch.Size([n, 1]))
y_val = 0.01 * y_rv.sample(torch.Size([n, 1]))


## Set hyperparameters for network

tau_list = []
tau = 0.5#/100. # iris 1/10
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

params_init = hamiltorch.util.flatten(net).to(device).clone()

step_size = 0.1 # 0.5
num_samples = 30000#1000
L = 20#3


def sample(beta):
    params_hmc = hamiltorch.sample_model(net, x_train, y_train,
                                         sampler=hamiltorch.Sampler.HMC_NUTS,
                                         params_init=params_init,
                                         num_samples=num_samples,
                                         step_size=step_size,
                                         num_steps_per_sample=L,
                                         burn=10000,
                                         tau_out=beta,
                                         tau_list=tau_list)

    sampled_params = np.asarray([t.numpy() for t in params_hmc[-1000:]])

    ax = sns.kdeplot(sampled_params, shade=True, cmap="PuBu")
    ax.patch.set_facecolor('white')
    ax.collections[0].set_alpha(0)
    ax.set_xlabel('$a prime$', fontsize=15)
    ax.set_ylabel('$b prime$', fontsize=15)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('beta {}'.format(beta))
    plt.show()

    pred_list, log_prob_list = hamiltorch.predict_model(net, x_train, y_train, samples=params_hmc[-1000:],
                                                        model_loss='regression', tau_out=1.,
                                                        tau_list=tau_list)

    nll = 0
    for output in pred_list:
        nll = nll + 0.5 * 1 * ((output - y_train) ** 2).sum(0)  # sum(0)

    print('\nExpected validation log probability: {:.2f}'.format(torch.stack(log_prob_list).mean()))
    plt.plot(log_prob_list)
    plt.show()

    return nll/pred_list.__len__()

def plot_energy(inverse_temp, avg_nlls_beta):

    # ordinary ls fit
    numbetas = inverse_temp.shape[0]
    design_x = np.vstack((np.ones(numbetas), 1 / inverse_temp)).T
    design_y = avg_nlls_beta
    design_y = design_y[:, np.newaxis]
    fit = inv(design_x.T.dot(design_x)).dot(design_x.T).dot(design_y)
    ols_intercept_estimate = fit[0][0]
    RLCT_estimate = fit[1][0]


    plt.scatter(1 / inverse_temp, avg_nlls_beta, label='nll beta')
    plt.plot(1 / inverse_temp, ols_intercept_estimate + RLCT_estimate / inverse_temp, 'b-', label='ols')
    if trueRLCT is not None:
        plt.title("d/2 = {} , true RLCT = {:.1f}, est RLCT = {:.1f}".format(w_dim/2, trueRLCT, RLCT_estimate), fontsize=8)
    # elif RLCT_ub is not None:
    #     plt.title("d/2 = {} , upper bound RLCT = {:.1f}, est RLCT = {:.1f}".format(args.w_dim/2, args.RLCT_ub, RLCT_estimate), fontsize=8)
    # else:
    #     plt.title("d/2 = {}, est RLCT = {:.1f}".format(args.w_dim/2, RLCT_estimate), fontsize=8)

    plt.xlabel("1/beta", fontsize=8)
    plt.ylabel("E^beta_w [nL_n(w)]", fontsize=8)
    plt.legend()
    plt.show()

    return RLCT_estimate


numbetas = 20
betasbegin = 0.8
betasend = 1.2
inverse_temp = np.flip(1 / np.linspace(1 / betasbegin, 1 / betasend, numbetas),axis=0)
avg_nlls_beta = np.empty(numbetas)


for beta_index in range(0, numbetas):
    beta = inverse_temp[beta_index]
    nll = sample(beta)
    avg_nlls_beta[beta_index] = nll

    if beta_index > 0:
        plot_energy(inverse_temp[0:beta_index + 1], avg_nlls_beta[0:beta_index + 1])

plot_energy(inverse_temp, avg_nlls_beta)
