import argparse
import os
import itertools
import time

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.distributions import Normal
import torch.multiprocessing
from torch.multiprocessing import Process, Manager

import matplotlib
matplotlib.use("Agg")
plt = matplotlib.pyplot
plt.rcParams["axes.titlesize"] = 8
import seaborn as sns
sns.set_style('white')

from utils import *
from models import *

from pyro.infer import MCMC, NUTS

# This is required both to get AMD CPUs to work well, but also
# to disable the aggressive multi-threading of the underlying
# linear algebra libraries, which interferes with our multiprocessing
# with PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = '' # disable CUDA
os.environ['MKL_DEBUG_CPU_TYPE'] = '5' # Get MKL to work properly on AMD CPU
os.environ['MKL_SERIAL'] = 'YES' # reduce thread usage in linalg
os.environ['OMP_NUM_THREADS'] = '1' # reduce thread usage in linalg


# dataset
def get_data(args):

    train_size = int(args.n)
    valid_size = int(args.n * 0.5)
    test_size = int(10000)

    X_rv = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(args.input_dim))
    y_rv = MultivariateNormal(torch.zeros(args.output_dim), torch.eye(args.output_dim))

    with torch.no_grad():

        X = X_rv.sample(torch.Size([train_size+valid_size]))
        X_test = args.X_test_std * X_rv.sample(torch.Size([test_size]))

        if args.realizable == 1:

            true_model = Model(args.input_dim, args.output_dim, args.ffrelu_layers, args.ffrelu_hidden, args.rr_hidden, args.use_rr_relu)
            true_model.eval()
            true_mean = true_model(X)
            true_mean_test = true_model(X_test)
            
        else:

            a = Normal(0.0, 1.0)
            a_params = 0.2 * a.sample((args.input_dim, args.rr_hidden))
            b = Normal(0.0, 1.0)
            b_params = 0.2 * b.sample((args.rr_hidden, args.output_dim))
            true_mean = torch.matmul(torch.matmul(X, a_params), b_params)
            true_mean_test = torch.matmul(torch.matmul(X_test, a_params), b_params)

        y = true_mean + y_rv.sample(torch.Size([train_size+valid_size]))
        y_test = true_mean_test + y_rv.sample(torch.Size([test_size]))

        dataset_train, dataset_valid = torch.utils.data.random_split(TensorDataset(X, y), [train_size, valid_size])
        dataset_test = TensorDataset(X_test, y_test)

        oracle_mse = (torch.norm(y_test - true_mean_test, dim=1)**2).mean()
        entropy = -torch.log((2 * np.pi) ** (-args.output_dim / 2) * torch.exp(-(1 / 2) * torch.norm(y_test - true_mean_test, dim=1) ** 2)).mean()

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=int(args.batchsize), shuffle=True)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=int(args.batchsize), shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=int(args.batchsize), shuffle=True)

    return train_loader, valid_loader, test_loader, oracle_mse, entropy


# model: small feedforward relu block, followed by reduced rank regression in last layers
class Model(nn.Module):

    def __init__(self, input_dim, output_dim, ffrelu_layers, ffrelu_hidden, rr_hidden, use_rr_relu):
        super(Model, self).__init__()

        self.use_rr_relu = use_rr_relu
        
        # feedforward relu block
        self.enc_sizes = np.concatenate(
            ([input_dim], np.repeat(ffrelu_hidden, ffrelu_layers + 1), [input_dim])).tolist()
        blocks = [[nn.Linear(in_f, out_f), nn.ReLU()]
                  for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        blocks = list(itertools.chain(*blocks))
        del blocks[-1]  # remove the last ReLu, don't need it in output layer
        self.feature_map = nn.Sequential(*blocks)

        # reduced rank regression block
        self.rr = nn.Sequential(
            nn.Linear(input_dim, rr_hidden, bias=False), # A
            nn.Linear(rr_hidden, output_dim, bias=False) # B
        )

        # reduced rank regression block with relu activation
        self.rr_relu = nn.Sequential(
            nn.Linear(input_dim, rr_hidden, bias=False), # A
            nn.ReLU(),
            nn.Linear(rr_hidden, output_dim, bias=False) # B
        ) 
        
    def forward(self, x):
        x = self.feature_map(x)
        if self.use_rr_relu == 1:
            return self.rr_relu(x)
        else:
            return self.rr(x)


def map_train(args, train_loader, valid_loader, test_loader, oracle_mse):

    model = Model(args.input_dim, args.output_dim, args.ffrelu_layers, args.ffrelu_hidden, args.rr_hidden, args.use_rr_relu)

    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=False, taskid=args.taskid)

    for it in range(args.train_epochs):

        model.train()
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            y_pred = model(data)
            l = (torch.norm(y_pred - target, dim=1)**2).mean()
            l.backward()
            opt.step()
            opt.zero_grad()
            running_loss += (torch.norm(y_pred - target, dim=1)**2).sum().detach().numpy()

        # between epochs
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for batch_idx, (data, target) in enumerate(valid_loader):
                valid_loss += (torch.norm(model(data) - target, dim=1)**2).sum()

        if args.use_early_stopping == 1:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # print test loss every now and then
        if it % args.log_interval == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for batch_idx, (data, target) in enumerate(test_loader):
                    ytest_pred = model(data)
                    test_loss += (torch.norm(ytest_pred - target, dim=1)**2).sum()
            print('MSE: train {:.3f}, validation {:.3f}, test {:.3f}, oracle on test set {:.3f}'.format(running_loss/len(train_loader.dataset), valid_loss/len(valid_loader.dataset), test_loss/len(test_loader.dataset), oracle_mse))

    return model


def laplace_last(model, args, X_train, Y_train, X_test, Y_test, lastlayeronly=False):

    A = list(model.parameters())[-2]
    A_map = A.view(-1).data.numpy()
    B = list(model.parameters())[-1]
    B_map = B.view(-1).data.numpy()
    
    if lastlayeronly:
        W_map = B_map
    else:
        W_map = np.concatenate((A_map, B_map))

    # get negative log posterior = negative log likelihood + negative log prior
    y_pred = model(X_train)
    nll = (torch.norm(y_pred - Y_train, dim=1)**2).mean()

    # Negative-log-prior
    nlp = 1 / 2 * A.flatten() @ (args.weight_decay * torch.eye(A.numel())) @ A.flatten() + 1 / 2 * B.flatten() @ (args.weight_decay * torch.eye(B.numel())) @ B.flatten()
    loss = nll + nlp

    if lastlayeronly:
        Lambda = exact_hessian(loss, [B])  # The Hessian of the negative log-posterior
    else:
        Lambda = exact_hessian(loss, [A, B])  # The Hessian of the negative log-posterior

    Sigma = torch.inverse(Lambda).detach().numpy()

    # posterior over w approximated as N(w_map, Sigma)
    sampled_weights = np.random.multivariate_normal(mean=W_map, cov=Sigma, size=args.R)

    with torch.no_grad():

        transformed_X_test = model.feature_map(X_test)

        if lastlayeronly:

            transformed_X_test = np.matmul(transformed_X_test, A.detach().numpy())
            
        pred_prob = 0

        for r in range(0, args.R):

            if lastlayeronly:
                sampled_b = sampled_weights[r,:].reshape(args.rr_hidden,args.output_dim)
                mean = np.matmul(transformed_X_test, sampled_b)
            else:
                sampled_a = sampled_weights[r,0:(args.rr_hidden*args.input_dim)].reshape(args.input_dim,args.rr_hidden)
                sampled_b = sampled_weights[r,-(args.rr_hidden*args.output_dim):].reshape(args.rr_hidden,args.output_dim)
                mean = np.matmul(np.matmul(transformed_X_test, sampled_a), sampled_b)
                
            pred_prob += (2 * np.pi) ** (-args.output_dim / 2) * torch.exp(-(1 / 2) * torch.norm(Y_test - mean, dim=1) ** 2)

    return -torch.log(pred_prob / args.R).mean()


def mcmc_last(model, args, X_train, Y_train, X_test, Y_test, lastlayeronly=False):
    
    B = list(model.parameters())[-1]
    A = list(model.parameters())[-2]
    
    transformed_X_train = model.feature_map(X_train)
    transformed_X_test = model.feature_map(X_test)

    if lastlayeronly:
        transformed_X_train = torch.matmul(transformed_X_train, A)
        transformed_X_test = torch.matmul(transformed_X_test, A)
        if args.use_rr_relu:
            transformed_X_train = torch.relu(transformed_X_train)
            transformed_X_test = torch.relu(transformed_X_test)

    kernel = NUTS(conditioned_pyro_rr, adapt_step_size=True)
    mcmc = MCMC(kernel, num_samples=args.R, warmup_steps=args.num_warmup, disable_progbar=True)
    if args.mcmc_prior_map == 1:
        mcmc.run(pyro_rr, transformed_X_train, Y_train, args.rr_hidden, beta=1.0, Bmap=B, Amap=A, relu=args.use_rr_relu, lastlayeronly=lastlayeronly)
    else:
        mcmc.run(pyro_rr, transformed_X_train, Y_train, args.rr_hidden, beta=1.0, Bmap=None, Amap=None, relu=args.use_rr_relu, lastlayeronly=lastlayeronly)
    sampled_weights = mcmc.get_samples()

    pred_prob = 0
    output_dim = Y_train.shape[1]
    for r in range(0, args.R):

        if lastlayeronly:
            mean = torch.matmul(transformed_X_test, sampled_weights['B'][r,:,:])
        else:
            if args.use_rr_relu:
                z = torch.relu(torch.matmul(transformed_X_test, sampled_weights['A'][r, :, :]))
            else:
                z = torch.matmul(transformed_X_test, sampled_weights['A'][r,:,:])
            mean = torch.matmul(z, sampled_weights['B'][r,:,:])

        pred_prob += (2 * np.pi) ** (-output_dim / 2) * torch.exp(-(1 / 2) * torch.norm(Y_test - mean, dim=1) ** 2)

    return -torch.log(pred_prob / args.R).mean()


def run_worker(i, n, G_mcmc_rrs, G_mcmc_lasts, G_maps, G_laplace_rrs, G_laplace_lasts, entropys, args):

    G_map = np.empty(args.MCs)
    G_mcmc_rr = np.empty(args.MCs)
    G_mcmc_last = np.empty(args.MCs)
    G_laplace_rr = np.empty(args.MCs)
    G_laplace_last = np.empty(args.MCs)
    entropy_array = np.empty(args.MCs)

    args.n = n
    if args.use_minibatch == 0:
        args.batchsize = n
    else:
        args.batchsize = 32

    start = time.time()
    
    for mc in range(0, args.MCs):

        train_loader, valid_loader, test_loader, oracle_mse, entropy = get_data(args)
        entropy_array[mc] = entropy

        X_train = train_loader.dataset[:][0]
        Y_train = train_loader.dataset[:][1]
        X_test = test_loader.dataset[:][0]
        Y_test = test_loader.dataset[:][1]

        model = map_train(args, train_loader, valid_loader, test_loader, oracle_mse)
        
        model.eval()
        
        G_map[mc] = -torch.log((2*np.pi)**(-args.output_dim /2) * torch.exp(-(1/2) * torch.norm(Y_test-model(X_test), dim=1)**2)).mean() - entropy

        G_laplace_rr[mc] = laplace_last(model, args, X_train, Y_train, X_test, Y_test, lastlayeronly=False) - entropy

        G_laplace_last[mc] = laplace_last(model, args, X_train, Y_train, X_test, Y_test, lastlayeronly=True) - entropy

        G_mcmc_rr[mc] = mcmc_last(model, args, X_train, Y_train, X_test, Y_test, lastlayeronly=False) - entropy
        
        G_mcmc_last[mc] = mcmc_last(model, args, X_train, Y_train, X_test, Y_test, lastlayeronly=True) - entropy

        print('[n = {}, mc {}] gen error: map {:.4f}, mcmc rr {:.4f}, laplace rr {:.4f}, mcmc last {:.4f}, laplace last {:.4f}'
              .format(n, mc, G_map[mc], G_mcmc_rr[mc], G_laplace_rr[mc], G_mcmc_last[mc], G_laplace_last[mc]))

    print('[n = {}] average gen error: MAP {}, mcmc rr {}, laplace rr {}, mcmc last {}, laplace last {}'
          .format(n, G_map.mean(), G_mcmc_rr.mean(), G_laplace_rr.mean(), G_mcmc_last.mean(), G_laplace_last.mean()))

    print('[n = {}] time taken(s): {}'.format(n, time.time() - start))

    G_mcmc_rrs[i] = G_mcmc_rr
    G_mcmc_lasts[i] = G_mcmc_last
    G_laplace_rrs[i] = G_laplace_rr
    G_laplace_lasts[i] = G_laplace_last
    G_maps[i] = G_map
    entropys[i] = entropy_array

    return


def main():

    parser = argparse.ArgumentParser(description='last layer Bayesian')

    parser.add_argument('--experiment-name', type=str, default='')

    parser.add_argument('--taskid', type=int, default=1)

    # Data
    parser.add_argument('--input-dim', type=int, default=3)

    parser.add_argument('--output-dim', type=int, default=3)

    parser.add_argument('--X-test-std', type=float, default=1.0)

    parser.add_argument('--realizable', type=int, default=0, help='1 if true distribution is realizable by model')


    # Model
    parser.add_argument('--ffrelu-layers',type=int, default=1, help='number of layers in feedforward relu block')

    parser.add_argument('--ffrelu-hidden',type=int, default=5, help='number of hidden units in feedforward relu block')

    parser.add_argument('--rr-hidden', type=int, default=3, help='number of hidden units in final reduced regression layers')

    parser.add_argument('--use-rr-relu', type=int, default=0, help='1 if true, 0 else')

    parser.add_argument('--use-minibatch', type=int, default=0, help='1 if use minbatch sgd for map training')

    parser.add_argument('--train-epochs', type=int, default=5000, help='number of epochs to find MAP')

    parser.add_argument('--use-early-stopping', type=int, default=0, help='1 to employ early stopping in map training based on validation loss')

    parser.add_argument('--log-interval', type=int, default=500, metavar='N', help='how many batches to wait before logging training status')

    parser.add_argument('--weight-decay', type=float, default=5e-4)

    # MCMC

    parser.add_argument('--mcmc-prior-map', type=int, default=0, help='1 if mcmc prior should be centered at map')
    
    parser.add_argument('--num-warmup', type=int, default=1000, help='burn in')

    parser.add_argument('--R', type=int, default=1000, help='number of MC draws from approximate posterior')

    parser.add_argument('--MCs', type=int, default=20, help='number of times to split into train-test')

    parser.add_argument('--num-n', type=int, default=10, help='number of sample sizes for learning curve')

    parser.add_argument('--seed', type=int, default=43)

    parser.add_argument('--cuda', action='store_true',default=False, help='flag for CUDA training')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.cuda = args.cuda and torch.cuda.is_available()

    init_model = Model(args.input_dim, args.output_dim, args.ffrelu_layers, args.ffrelu_hidden, args.rr_hidden, args.use_rr_relu)
    args.total_param_count = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
    args.w_dim = args.rr_hidden*(args.input_dim + args.output_dim) # number of parameters in reduced rank regression layers
    H0 = min(args.input_dim, args.output_dim, args.rr_hidden)
    args.trueRLCT = theoretical_RLCT('rr', (args.input_dim, args.output_dim, H0, args.rr_hidden))

    n_range = np.rint(np.logspace(2.3, 3.0, 10)).astype(int)
    args.n_range = n_range

    print(args)
    if not os.path.exists('lastlayersims/'):
        os.mkdir('lastlayersims/')

    torch.save(args,'lastlayersims/{}_taskid{}_args.pt'.format(args.experiment_name, args.taskid))


    # We do each n in parallel
    manager = Manager()

    m_G_maps = manager.list(n_range)
    m_G_mcmc_rrs = manager.list(n_range)
    m_G_mcmc_lasts = manager.list(n_range)
    m_G_laplace_rrs = manager.list(n_range)
    m_G_laplace_lasts = manager.list(n_range)
    m_entropys = manager.list(n_range)

    jobs = []
    
    for i in range(len(n_range)):
        n = n_range[i]
        print("Starting job [n = {0}]".format(n))
        p = Process(target=run_worker, args=(i, n, m_G_mcmc_rrs, m_G_mcmc_lasts, m_G_maps, m_G_laplace_rrs, m_G_laplace_lasts, m_entropys, args))

        jobs.append(p)
        p.start()

    # block on all jobs completing
    for p in jobs:
        p.join()

    # variables to save for producing graphics/table later
    G_mcmc_rrs = list(m_G_mcmc_rrs)
    G_mcmc_lasts = list(m_G_mcmc_lasts)
    G_maps = list(m_G_maps)
    G_laplace_rrs = list(m_G_laplace_rrs)
    G_laplace_lasts = list(m_G_laplace_lasts)
    entropys = list(m_entropys)

    results = dict()
    results['mcmc_rr'] = G_mcmc_rrs
    results['mcmc_last'] = G_mcmc_lasts
    results['map'] = G_maps
    results['laplace_rr'] = G_laplace_rrs
    results['laplace_last'] = G_laplace_lasts
    results['entropy'] = entropys
    torch.save(results,'lastlayersims/{}_taskid{}_results.pt'.format(args.experiment_name, args.taskid))


if __name__ == "__main__":
    main()
