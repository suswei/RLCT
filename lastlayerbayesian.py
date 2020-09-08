# wiseodd/last_layer_laplace

import matplotlib
matplotlib.use("Agg")
from torch.distributions.multivariate_normal import MultivariateNormal

import seaborn as sns
sns.set_style('white')

from torch.utils.data import TensorDataset

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

    X_rv = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(args.input_dim))
    y_rv = MultivariateNormal(torch.zeros(args.output_dim), torch.eye(args.output_dim))

    with torch.no_grad():

        X = X_rv.sample(torch.Size([train_size+valid_size]))
        X_test = args.X_test_std * X_rv.sample(torch.Size([test_size]))

        if args.realizable:

            true_model = Model(args.input_dim, args.output_dim, args.ffrelu_layers, args.ffrelu_hidden, args.rr_hidden)
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

        dataset_train, dataset_valid = torch.utils.data.random_split(TensorDataset(X, y),[train_size, valid_size])
        dataset_test = TensorDataset(X_test, y_test)

        oracle_mse = (torch.norm(y_test - true_mean_test, dim=1)**2).mean()
        entropy = -torch.log((2 * np.pi) ** (-args.output_dim / 2) * torch.exp(-(1 / 2) * torch.norm(y_test - true_mean_test, dim=1) ** 2)).mean()

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=int(args.batchsize), shuffle=True)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=int(args.batchsize), shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=int(args.batchsize), shuffle=True)

    return train_loader, valid_loader, test_loader, oracle_mse, entropy


# model: small feedforward relu block, followed by reduced rank regression in last layers
class Model(nn.Module):

    def __init__(self, input_dim, output_dim, ffrelu_layers, ffrelu_hidden, rr_hidden):
        super(Model, self).__init__()

        self.enc_sizes = np.concatenate(
            ([input_dim], np.repeat(ffrelu_hidden, ffrelu_layers + 1), [input_dim])).tolist()
        blocks = [[nn.Linear(in_f, out_f), nn.ReLU()]
                  for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        blocks = list(itertools.chain(*blocks))
        del blocks[-1]  # remove the last ReLu, don't need it in output layer

        self.feature_map = nn.Sequential(*blocks)

        # # TODO: variable layers
        # self.feature_map = nn.Sequential(
        #     nn.Linear(input_dim, ffrelu_hidden),
        #     nn.ReLU(),
        #     nn.Linear(ffrelu_hidden, ffrelu_hidden),
        #     nn.ReLU(),
        #     nn.Linear(ffrelu_hidden, input_dim),
        # )

        # TODO: linear layer alternative?
        self.rr = nn.Sequential(
            nn.Linear(input_dim, rr_hidden, bias=False), # A
            nn.Linear(rr_hidden, output_dim, bias=False) # B
        )



    def forward(self, x):
        x = self.feature_map(x)
        return self.rr(x)


def map_train(args, train_loader, valid_loader, test_loader, oracle_mse):

    model = Model(args.input_dim, args.output_dim, args.ffrelu_layers, args.ffrelu_hidden, args.rr_hidden)

    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=False, taskid=args.taskid)

    for it in range(args.train_epochs):

        model.train()
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            y_pred = model(data).squeeze()
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
                valid_loss += (torch.norm(model(data).squeeze() - target, dim=1)**2).sum()

        if args.early_stopping:
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
                    ytest_pred = model(data).squeeze()
                    test_loss += (torch.norm(ytest_pred - target, dim=1)**2).sum()
            print('MSE: train {:.3f}, validation {:.3f}, test {:.3f}, oracle on test set {:.3f}'.format(running_loss/len(train_loader.dataset), valid_loss/len(valid_loader.dataset), test_loss/len(test_loader.dataset), oracle_mse))

    return model


def lastlayerlaplace(model, args, X_train, Y_train, X_test, Y_test):

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


def lastlayermcmc(model, args, X_train, Y_train, X_test, Y_test):
    
    transformed_X_train = model.feature_map(X_train)
    transformed_X_test = model.feature_map(X_test)

    beta = 1.0

    kernel = NUTS(conditioned_pyro_rr, adapt_step_size=True)
    mcmc = MCMC(kernel, num_samples=args.R, warmup_steps=args.num_warmup, disable_progbar=True)
    mcmc.run(pyro_rr, transformed_X_train, Y_train, args.rr_hidden, beta)
    sampled_weights = mcmc.get_samples()

    pred_prob = 0
    output_dim = Y_train.shape[1]
    for r in range(0, args.R):
        mean = torch.matmul(torch.matmul(transformed_X_test, sampled_weights['a'][r,:,:]), sampled_weights['b'][r,:,:])
        pred_prob += (2 * np.pi) ** (-output_dim / 2) * torch.exp(-(1 / 2) * torch.norm(Y_test - mean, dim=1) ** 2)

    return -torch.log(pred_prob / args.R).mean()


def run_worker(i, n, avg_G_llb, std_G_llb, avg_G_map, std_G_map, avg_entropy, std_entropy, args):

    G_map = np.empty(args.MCs)
    G_llb = np.empty(args.MCs)
    G_lll = np.empty(args.MCs)
    entropy_array = np.empty(args.MCs)

    args.n = n
    if args.minibatch == 0:
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

        # G_lll[mc] = lastlayerlaplace(model, args, X_train, Y_train, X_test, Y_test) - entropy

        G_llb[mc] = lastlayermcmc(model, args, X_train, Y_train, X_test, Y_test) - entropy

        print('[n = {}] mc {}, gen error: map {}, last layer mcmc {}'
              .format(n, mc, G_map[mc], G_llb[mc]))
        # print('[n = {}] mc {}, gen error: map {}, last layer mcmc {}, last layer laplace {}'
        #       .format(n, mc, G_map[mc], G_llb[mc], G_lll[mc]))

    print('[n = {}] average gen error: MAP {}, last layer mcmc {}'
          .format(n, G_map.mean(), G_llb.mean()))

    print('[n = {}] time taken(s): {}'.format(n, time.time() - start))
    
    avg_G_llb[i] = G_llb.mean()
    std_G_llb[i] = G_llb.std()
    avg_G_map[i] = G_map.mean()
    std_G_map[i] = G_map.std()
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

    parser.add_argument('--realizable', type=int, default=0, help='1 if true distribution is realizable by model')


    # Model
    parser.add_argument('--ffrelu-layers',type=int, default=1, help='number of layers in feedforward relu block')

    parser.add_argument('--ffrelu-hidden',type=int, default=5, help='number of hidden units in feedforward relu block')

    parser.add_argument('--rr-hidden', type=int, default=3, help='number of hidden units in final reduced regression layers')

    parser.add_argument('--minibatch', type=int, default=0, help='1 if use minbatch sgd for map training')

    parser.add_argument('--train-epochs', type=int, default=5000, help='number of epochs to find MAP')

    parser.add_argument('--early-stopping', type=int, default=0, help='1 to employ early stopping in map training based on validation loss')

    parser.add_argument('--log-interval', type=int, default=500, metavar='N', help='how many batches to wait before logging training status')

    parser.add_argument('--weight-decay', type=float, default=5e-4)

    # MCMC

    parser.add_argument('--num-warmup', type=int, default=10000, help='burn in')

    parser.add_argument('--R', type=int, default=1000, help='number of MC draws from approximate posterior')

    parser.add_argument('--MCs', type=int, default=20, help='number of times to split into train-test')

    parser.add_argument('--num-n', type=int, default=10, help='number of sample sizes')

    parser.add_argument('--seed', type=int, default=43)

    parser.add_argument('--cuda', action='store_true',default=False, help='flag for CUDA training')

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


    init_model = Model(args.input_dim, args.output_dim, args.ffrelu_layers, args.ffrelu_hidden, args.rr_hidden)
    args.total_param_count = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
    args.w_dim = args.rr_hidden*(args.input_dim + args.output_dim) # number of parameters in reduced rank regression layers
    H0 = min(args.input_dim, args.output_dim, args.rr_hidden)
    args.trueRLCT = theoretical_RLCT('rr', (args.input_dim, args.output_dim, H0, args.rr_hidden))
    print(args)

    n_range = np.rint(1/np.linspace(1/200, 1/1000, args.num_n)).astype(int)

    # We do each n in parallel
    manager = Manager()
    m_avg_G_llb = manager.list(n_range)
    m_std_G_llb = manager.list(n_range)
    m_avg_G_map = manager.list(n_range)
    m_std_G_map = manager.list(n_range)
    m_avg_entropy = manager.list(n_range)
    m_std_entropy = manager.list(n_range)
    
    jobs = []
    
    for i in range(len(n_range)):
        n = n_range[i]
        print("Starting job [n = {0}]".format(n))
        p = Process(target=run_worker, args=(i, n, m_avg_G_llb, m_std_G_llb,
                                            m_avg_G_map, m_std_G_map, m_avg_entropy, 
                                            m_std_entropy, args))
        jobs.append(p)
        p.start()

    # block on all jobs completing
    for p in jobs:
        p.join()
    
    # Convert the managed shared lists into numpy arrays
    avg_G_llb = np.array(m_avg_G_llb)
    std_G_llb = np.array(m_std_G_llb)
    avg_G_map = np.array(m_avg_G_map)
    std_G_map = np.array(m_std_G_map)
    avg_entropy = np.array(m_avg_entropy)
    std_entropy = np.array(m_std_entropy)
    
    print('avg LLB gen err {}, std {}'.format(avg_G_llb, std_G_llb))
    print('avg MAP gen err {}, std {}'.format(avg_G_map, std_G_map))

    # varaibles to save for producing graphics/table later
    save_objects = (n_range, avg_G_llb,std_G_llb, avg_G_map, std_G_map)

    # summarize results
    if args.realizable:
        ols_map = OLS(avg_G_map, 1 / n_range).fit()
        map_slope = ols_map.params[0]

        ols_llb = OLS(avg_G_llb, 1 / n_range).fit()
        llb_intercept = 0.0
        llb_slope = ols_llb.params[0]
    else:
        ols_map = OLS(avg_G_map, add_constant(1 / n_range)).fit()
        map_slope = ols_map.params[1]

        ols_llb = OLS(avg_G_llb, add_constant(1 / n_range)).fit()
        llb_intercept = ols_llb.params[0]
        llb_slope = ols_llb.params[1]

    print('estimated RLCT {}'.format(llb_slope))

    # learning curves
    fig, ax = plt.subplots()
    ax.errorbar(1/n_range, avg_G_llb, yerr=std_G_llb, fmt='-o', c='r', label='En G(n) for last layer mcmc')
    ax.errorbar(1/n_range, avg_G_map, yerr=std_G_map, fmt='-o', c='g', label='En G(n) for MAP')
    plt.plot(1 / n_range, llb_intercept + llb_slope / n_range, 'r--', label='ols fit for last layer mcmc')
    plt.xlabel('1/n')
    plt.title('map slope {:.2f}, parameter count {}, llb slope {:.2f}, true RLCT {}'.format(map_slope, args.total_param_count, llb_slope, args.trueRLCT))
    plt.legend()
    plt.savefig('taskid{}.png'.format(args.taskid))
    # DM: plt.show()

    torch.save(args,'taskid{}_args.pt'.format(args.taskid))
    torch.save(save_objects,'taskid{}_results.pt'.format(args.taskid))


if __name__ == "__main__":
    main()
