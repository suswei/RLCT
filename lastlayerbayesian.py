# wiseodd/last_layer_laplace

import time

import matplotlib
matplotlib.use("Agg")
from torch.distributions.multivariate_normal import MultivariateNormal

import seaborn as sns
sns.set_style('white')

from torch.utils.data import TensorDataset
from torch import Tensor

from main import *
from utils import exact_hessian

plt = matplotlib.pyplot

# DM
import torch.multiprocessing
from torch.multiprocessing import Process, Manager

# DM
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

    a = Normal(0.0, 1.0)
    a_params = 0.2 * a.sample((args.input_dim, args.rr_hidden))
    b = Normal(0.0, 1.0)
    b_params = 0.2 * b.sample((args.rr_hidden, args.output_dim))

    X_rv = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(args.input_dim))
    y_rv = MultivariateNormal(torch.zeros(args.output_dim), torch.eye(args.output_dim))
    true_model = Model(args.input_dim, args.output_dim, args.ffrelu_hidden, args.rr_hidden)
    true_model.eval()

    with torch.no_grad():
        # training +valid data
        X = X_rv.sample(torch.Size([train_size+valid_size]))
        if args.realizable:
            true_mean = true_model(X)
        else:
            true_mean = torch.matmul(torch.matmul(X, a_params), b_params)
        y = true_mean + y_rv.sample(torch.Size([train_size+valid_size]))
        dataset_train, dataset_valid = torch.utils.data.random_split(TensorDataset(X, y),[train_size,valid_size])

        # testing data
        X = args.X_test_std * X_rv.sample(torch.Size([test_size]))
        if args.realizable:
            true_mean = true_model(X)
        else:
            true_mean = torch.matmul(torch.matmul(X, a_params), b_params)
        y = true_mean + y_rv.sample(torch.Size([test_size]))
        dataset_test = TensorDataset(X, y)
        oracle_mse = (torch.norm(y - true_mean, dim=1)**2).mean()
        entropy = -torch.log((2 * np.pi) ** (-args.output_dim / 2) * torch.exp(
            -(1 / 2) * torch.norm(y - true_mean, dim=1) ** 2)).mean()

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True)

    return train_loader, valid_loader, test_loader, oracle_mse, entropy


# model: small feedforward relu block, followed by reduced rank regression in last layers
class Model(nn.Module):

    def __init__(self, input_dim, output_dim, ffrelu_hidden, rr_hidden):
        super(Model, self).__init__()

        # TODO: variable layers
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, ffrelu_hidden),
            nn.ReLU(),
            nn.Linear(ffrelu_hidden, ffrelu_hidden),
            nn.ReLU(),
            nn.Linear(ffrelu_hidden, input_dim),
        )

        # TODO: linear layer alternative?
        self.rr = nn.Sequential(
            nn.Linear(input_dim, rr_hidden, bias=False), # A
            nn.Linear(rr_hidden, output_dim, bias=False) # B
        )

    def forward(self, x):
        x = self.feature_map(x)
        return self.rr(x)


def map_train(args, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, oracle_mse):

    model = Model(args.input_dim, args.output_dim, args.ffrelu_hidden, args.rr_hidden)
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=False, taskid=args.taskid)

    # TODO: is it necessary to implement mini batch SGD?
    for it in range(args.train_epochs):
        model.train()
        y_pred = model(X_train).squeeze()
        l = (torch.norm(y_pred - Y_train, dim=1)**2).mean()
        l.backward()
        opt.step()
        opt.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = (torch.norm(model(X_valid).squeeze() - Y_valid, dim=1)**2).mean()
            
        if it % 100 == 0:
            model.eval()
            ytest_pred = model(X_test).squeeze()
            test_loss = (torch.norm(ytest_pred - Y_test, dim=1)**2).mean()
            print('MSE: train {:.3f}, validation {:.3f}, test {:.3f}, oracle on test set {:.3f}'.format(l.item(), valid_loss, test_loss.item(), oracle_mse))

        if args.early_stopping:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return model

def LastLayerLaplace(model, args, X_train, Y_train, X_valid, Y_valid, X_test, Y_test):

    transformed_X_test = model.feature_map(X_test)

    A = list(model.parameters())[-2]
    A_map = A.view(-1).data.numpy()
    B = list(model.parameters())[-1]
    B_map = B.view(-1).data.numpy()

    W_map = np.concatenate((A_map, B_map))

    # get negative log posterior = negative log likelihood + negative log prior
    y_pred = model(X_train).squeeze()
    nll = (torch.norm(y_pred - Y_train, dim=1)**2).mean()
    # Negative-log-prior
    nlp = 1 / 2 * A.flatten() @ (args.weight_decay * torch.eye(A.numel())) @ A.flatten() + 1 / 2 * B.flatten() @ (args.weight_decay * torch.eye(B.numel())) @ B.flatten()
    loss = nll + nlp

    Lambda = exact_hessian(loss, [A,B])  # The Hessian of the negative log-posterior
    Sigma = torch.inverse(Lambda).detach().numpy()

    # posterior over w approximated as N(w_map, Sigma)
    sampled_weights = np.random.multivariate_normal(mean=W_map, cov=Sigma, size=args.R)

    with torch.no_grad():

        pred_prob = 0
        for r in range(0, args.R):
            sampled_a = sampled_weights[r,0:(args.rr_hidden*args.input_dim)].reshape(args.input_dim,args.rr_hidden)
            sampled_b = sampled_weights[r,-(args.rr_hidden*args.output_dim):].reshape(args.rr_hidden,args.output_dim)

            mean = np.matmul(np.matmul(transformed_X_test.detach().numpy(),sampled_a),sampled_b)
            pred_prob += (2 * np.pi) ** (-args.output_dim / 2) * torch.exp(-(1 / 2) * torch.norm(Y_test - mean, dim=1) ** 2)

    return -torch.log(pred_prob / args.R).mean()


def lastlayer_approxinf(model, args, X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    
    transformed_X_train = model.feature_map(X_train)
    transformed_X_valid = model.feature_map(X_valid)
    transformed_X_test = model.feature_map(X_test)

    transformed_train_loader = torch.utils.data.DataLoader(
        TensorDataset(Tensor(transformed_X_train), torch.as_tensor(Y_train, dtype=torch.long)),
        batch_size=args.batchsize, shuffle=True)
    transformed_valid_loader = torch.utils.data.DataLoader(
        TensorDataset(Tensor(transformed_X_valid), torch.as_tensor(Y_valid, dtype=torch.long)), 
        batch_size=args.batchsize, shuffle=True)

    if args.posterior_method == 'ivi':

        # parameters for train_implicitVI
        mc = 1
        beta_index = 0
        args.betas = [1.0]
        saveimgpath = None
        args.dataset = 'reducedrank_synthetic'
        args.H = args.rr_hidden

        # TODO: strip train_implicitVI to simplest possible inputs
        G = train_implicitVI(transformed_train_loader, transformed_valid_loader, args, mc, beta_index, saveimgpath)
        args.epsilon_dim = args.rr_hidden * (args.input_dim + args.output_dim)
        eps = torch.randn(args.R, args.epsilon_dim)
        sampled_weights = G(eps)
        list_of_param_dicts = weights_to_dict(args, sampled_weights)

        pred_prob = 0
        output_dim = transformed_X_test.shape[1]
        for param_dict in list_of_param_dicts:
            mean = torch.matmul(torch.matmul(transformed_X_test, param_dict['a']), param_dict['b'])
            pred_prob += (2 * np.pi) ** (-output_dim / 2) * torch.exp(-(1 / 2) * torch.norm(Y_test - mean, dim=1) ** 2)

    elif args.posterior_method == 'mcmc':

        wholex = transformed_train_loader.dataset[:][0]
        wholey = transformed_train_loader.dataset[:][1]
        beta = 1.0

        kernel = NUTS(conditioned_pyro_rr, adapt_step_size=True)
        mcmc = MCMC(kernel, num_samples=args.R, warmup_steps=args.num_warmup, disable_progbar=True)
        mcmc.run(pyro_rr, wholex, wholey, args.rr_hidden, beta)
        sampled_weights = mcmc.get_samples()

        pred_prob = 0
        output_dim = wholey.shape[1]
        for r in range(0, args.R):
            mean = torch.matmul(torch.matmul(transformed_X_test, sampled_weights['a'][r,:,:]), sampled_weights['b'][r,:,:])
            pred_prob += (2 * np.pi) ** (-output_dim / 2) * torch.exp(-(1 / 2) * torch.norm(Y_test - mean, dim=1) ** 2)

    return -torch.log(pred_prob / args.R).mean()


def run_worker(i, n, avg_llb_gen_err, std_llb_gen_err,
                    avg_map_gen_err, std_map_gen_err, avg_entropy, std_entropy, args):
    map_gen_err = np.empty(args.MCs)
    llb_gen_err = np.empty(args.MCs)
    entropy_array = np.empty(args.MCs)

    args.n = n

    start = time.time()
    
    for mc in range(0, args.MCs):
        train_loader, valid_loader, test_loader, oracle_mse, entropy = get_data(args)
        entropy_array[mc] = entropy

        X_train = train_loader.dataset[:][0]
        Y_train = train_loader.dataset[:][1]
        X_valid = valid_loader.dataset[:][0]
        Y_valid = valid_loader.dataset[:][1]
        X_test = test_loader.dataset[:][0]
        Y_test = test_loader.dataset[:][1]

        model = map_train(args, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, oracle_mse)
        
        model.eval()
        map_gen_err[mc] = -torch.log((2*np.pi)**(-args.output_dim /2) * torch.exp(-(1/2) * torch.norm(Y_test-model(X_test), dim=1)**2)).mean() - entropy

        llb_gen_err[mc] = lastlayer_approxinf(model, args, X_train, Y_train, X_valid, Y_valid, X_test, Y_test) - entropy

        print('[n = {}] mc {}, gen error: map {}, bayes last layer {}'
              .format(n, mc, map_gen_err[mc], llb_gen_err[mc]))


    print('[n = {}] average gen error: MAP {}, bayes {}'
          .format(n, map_gen_err.mean(), llb_gen_err.mean()))
    print('[n = {}] time taken(s): {}'.format(n, time.time() - start))
    
    avg_llb_gen_err[i] = llb_gen_err.mean()
    std_llb_gen_err[i] = llb_gen_err.std()
    avg_map_gen_err[i] = map_gen_err.mean()
    std_map_gen_err[i] = map_gen_err.std()
    avg_entropy[i] = entropy_array.mean()
    std_entropy[i] = entropy_array.std()
    return
       
def main():

    parser = argparse.ArgumentParser(description='last layer Bayesian')

    parser.add_argument('--taskid', type=int, default=1)

    # Data
    parser.add_argument('--input-dim', type=int, default=3)

    parser.add_argument('--output-dim', type=int, default=3)

    parser.add_argument('--X-test-std', type=float, default=1.0)

    parser.add_argument('--realizable', type=int, default=0)

    # Model
    parser.add_argument('--ffrelu-hidden',type=int,default=5, help='number of hidden units in feedforward relu layers')

    parser.add_argument('--rr-hidden', type=int, default=3, help='number of hidden units in final reduced regression layers')

    parser.add_argument('--early-stopping', type=int, default=0)

    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--weight-decay', type=float, default=5e-4)

    # posterior method
    parser.add_argument('--posterior_method', type=str, default='mcmc',choices=['mcmc','ivi'])

    # MCMC
    parser.add_argument('--num-warmup', type=int, default=10000, help='burn in')

    # IVI

    parser.add_argument('--batchsize', type=int, default=50, help='used in IVI')

    parser.add_argument('--epsilon_mc', type=int, default=100, help='used in IVI')

    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--pretrainDepochs', type=int, default=100,
                        help='number of epochs to pretrain discriminator')

    parser.add_argument('--trainDepochs', type=int, default=20,
                        help='number of epochs to train discriminator for each minibatch update of generator')

    parser.add_argument('--n_hidden_D', type=int, default=128,
                        help='number of hidden units in discriminator D')

    parser.add_argument('--num_hidden_layers_D', type=int, default=1,
                        help='number of hidden layers in discriminatror D')

    parser.add_argument('--n_hidden_G', type=int, default=128,
                        help='number of hidden units in generator G')

    parser.add_argument('--num_hidden_layers_G', type=int, default=1,
                        help='number of hidden layers in generator G')

    parser.add_argument('--lr_primal', type=float,  default=0.01, metavar='LR',
                        help='primal learning rate (default: 0.01)')

    parser.add_argument('--lr_dual', type=float, default=0.001, metavar='LR',
                        help='dual learning rate (default: 0.01)')

    # averaging
    parser.add_argument('--MCs', type=int, default=20,
                        help='number of times to split into train-test')

    parser.add_argument('--R', type=int, default=1000,
                        help='number of MC draws from approximate posterior')

    parser.add_argument('--num-n', type=int, default=10,
                        help='number of sample sizes')


    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--cuda', action='store_true',default=False,
                        help='flag for CUDA training')

    parser.add_argument('--train-epochs', type=int, default=5000,
                        help='number of epochs to find MAP')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.cuda = args.cuda and torch.cuda.is_available()

    # change to boolean
    if args.early_stopping == 0:
        args.early_stopping = False
    else:
        args.early_stopping = True
    if args.realizable == 0:
        args.realizable = False
    else:
        args.realizable = True

    # TODO: w_dim and total_param_count depend on model and shouldn't be hardcoded as follows
    args.w_dim = args.rr_hidden*(args.input_dim + args.output_dim)
    args.total_param_count = (args.input_dim + args.rr_hidden + args.input_dim) * args.rr_hidden + args.w_dim
    H0 = min(args.input_dim,args.output_dim,args.rr_hidden)
    args.trueRLCT = theoretical_RLCT('rr', (args.input_dim, args.output_dim, H0, args.rr_hidden))
    print(args)

    n_range = np.round(1/np.linspace(1/200, 1/10000, args.num_n))


    # We do each n in parallel
    manager = Manager()
    m_avg_llb_gen_err = manager.list(n_range)
    m_std_llb_gen_err = manager.list(n_range)
    m_avg_map_gen_err = manager.list(n_range)
    m_std_map_gen_err = manager.list(n_range)
    m_avg_entropy = manager.list(n_range)
    m_std_entropy = manager.list(n_range)
    
    jobs = []
    
    for i in range(len(n_range)):
        n = n_range[i]
        print("Starting job [n = {0}]".format(n))
        p = Process(target=run_worker, args=(i, n, m_avg_llb_gen_err, m_std_llb_gen_err,
                                            m_avg_map_gen_err, m_std_map_gen_err, m_avg_entropy, 
                                            m_std_entropy, args))
        jobs.append(p)
        p.start()

    # block on all jobs completing
    for p in jobs:
        p.join()
    
    # Convert the managed shared lists into numpy arrays
    avg_llb_gen_err = np.array(m_avg_llb_gen_err)
    std_llb_gen_err = np.array(m_std_llb_gen_err)
    avg_map_gen_err = np.array(m_avg_map_gen_err)
    std_map_gen_err = np.array(m_std_map_gen_err)
    avg_entropy = np.array(m_avg_entropy)
    std_entropy = np.array(m_std_entropy)
    
    print('avg LLB gen err {}, std {}'.format(avg_llb_gen_err, std_llb_gen_err))
    print('avg MAP gen err {}, std {}'.format(avg_map_gen_err, std_map_gen_err))

    # varaibles to save for producing graphics/table later
    save_objects = (n_range, avg_llb_gen_err,std_llb_gen_err, avg_map_gen_err, std_map_gen_err)

    # summarize results
    if args.realizable:
        ols_map = OLS(avg_map_gen_err, 1 / n_range).fit()
        map_slope = ols_map.params[0]

        ols_llb = OLS(avg_llb_gen_err, 1 / n_range).fit()
        llb_intercept = 0.0
        llb_slope = ols_llb.params[0]
    else:
        ols_map = OLS(avg_map_gen_err, add_constant(1 / n_range)).fit()
        map_slope = ols_map.params[1]

        ols_llb = OLS(avg_llb_gen_err, add_constant(1 / n_range)).fit()
        llb_intercept = ols_llb.params[0]
        llb_slope = ols_llb.params[1]

    print('estimated RLCT {}'.format(llb_slope))

    # learning curves
    fig, ax = plt.subplots()
    ax.errorbar(1/n_range, avg_llb_gen_err, yerr=std_llb_gen_err, fmt='-o', c='r', label='En G(n) for last layer Bayes predictive')
    ax.errorbar(1/n_range, avg_map_gen_err, yerr=std_map_gen_err, fmt='-o', c='g', label='En G(n) for MAP')
    plt.plot(1 / n_range, llb_intercept + llb_slope / n_range, 'r--', label='ols fit for last-layer-Bayes')
    plt.xlabel('1/n')
    plt.title('map slope {:.2f}, parameter count {}, LLB slope {:.2f}, true RLCT {}'.format(map_slope, args.total_param_count, llb_slope, args.trueRLCT))
    plt.legend()
    plt.savefig('taskid{}.png'.format(args.taskid))
    # DM: plt.show()

    torch.save(args,'taskid{}_args.pt'.format(args.taskid))
    torch.save(save_objects,'taskid{}_results.pt'.format(args.taskid))

if __name__ == "__main__":
    main()
