# wiseodd/last_layer_laplace

import matplotlib
from torch.distributions.multivariate_normal import MultivariateNormal

import seaborn as sns; sns.set_style('white')

from torch.utils.data import TensorDataset
from torch import Tensor

from main import *

plt = matplotlib.pyplot


# dataset
def get_data(args):

    train_size = args.n
    valid_size = int(args.n * 0.5)
    test_size = 10000

    a = Normal(0.0, 1.0)
    a_params = 0.2 * a.sample((args.input_dim, args.H))
    b = Normal(0.0, 1.0)
    b_params = 0.2 * b.sample((args.H, args.output_dim))
    X_rv = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(args.input_dim))
    y_rv = MultivariateNormal(torch.zeros(args.output_dim), torch.eye(args.output_dim))

    # training +valid data
    X = X_rv.sample(torch.Size([train_size+valid_size]))
    true_mean = torch.matmul(torch.matmul(X, a_params), b_params)
    y = true_mean + y_rv.sample(torch.Size([train_size+valid_size]))
    dataset_train, dataset_valid = torch.utils.data.random_split(TensorDataset(X, y),[train_size,valid_size])

    # testing data -- change X distribution?
    X = X_rv.sample(torch.Size([test_size]))
    true_mean = torch.matmul(torch.matmul(X, a_params), b_params)
    y = true_mean + y_rv.sample(torch.Size([test_size]))
    dataset_test = TensorDataset(X, y)
    baseline = (torch.norm(y - true_mean, dim=1)**2).mean()
    gen_err_baseline = -torch.log((2 * np.pi) ** (-args.output_dim / 2) * torch.exp(
        -(1 / 2) * torch.norm(y - true_mean, dim=1) ** 2)).mean()

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True)

    return train_loader, valid_loader, test_loader, baseline, gen_err_baseline


# model
class Model(nn.Module):

    def __init__(self, input_dim, output_dim, h, rr_H):
        super(Model, self).__init__()

        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, h),
            # nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            # nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, input_dim),
        )

        self.clf = nn.Sequential(
            nn.Linear(input_dim, rr_H, bias=False),
            nn.Linear(rr_H, output_dim, bias=False)
            # nn.Linear(input_dim,output_dim,bias=False)
        )

    def forward(self, x):
        x = self.feature_map(x)
        return self.clf(x)


# train module
def map_train(args, X_train, Y_train, X_test, Y_test, baseline):

    n, input_dim = X_train.shape
    n, output_dim = Y_train.shape

    model = Model(input_dim, output_dim, args.feature_map_hidden, args.H)
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    for it in range(5000):
        model.train()
        y_pred = model(X_train).squeeze()
        l = (torch.norm(y_pred - Y_train, dim=1)**2).mean()
        l.backward()
        opt.step()
        opt.zero_grad()

        if it % 100 == 0:
            model.eval()
            ytest_pred = model(X_test).squeeze()
            test_loss = (torch.norm(ytest_pred - Y_test, dim=1)**2).mean()
            print('negative log prob loss: train {:.3f}, test {:.3f}, baseline {:.3f}'.format(l.item(), test_loss.item(),baseline))

    return model


def posterior_sample(args, train_loader, test_loader, transformed_X_test, Y_test):

    mc = 1
    beta_index = 0
    args.betas = [1.0]
    saveimgpath = None

    if args.posterior_method == 'ivi':
        G = train_implicitVI(train_loader, test_loader, args, mc, beta_index, saveimgpath)
        eps = torch.randn(args.R, args.epsilon_dim)
        sampled_weights = G(eps)
        list_of_param_dicts = weights_to_dict(args, sampled_weights)

        pred_prob = 0
        output_dim = transformed_X_test.shape[1]
        for param_dict in list_of_param_dicts:
            mean = torch.matmul(torch.matmul(transformed_X_test, param_dict['a']), param_dict['b'])
            pred_prob += (2 * np.pi) ** (-output_dim / 2) * torch.exp(-(1 / 2) * torch.norm(Y_test - mean, dim=1) ** 2)

        return -torch.log(pred_prob / args.R).mean()

    elif args.posterior_method == 'mcmc':

        wholex = train_loader.dataset[:][0]
        wholey = train_loader.dataset[:][1]
        beta = 1.0

        kernel = NUTS(conditioned_pyro_rr, adapt_step_size=True)
        mcmc = MCMC(kernel, num_samples=args.R, warmup_steps=args.num_warmup)
        mcmc.run(pyro_rr, wholex, wholey, args.H, beta)
        sampled_weights = mcmc.get_samples()

        pred_prob = 0
        output_dim = wholey.shape[1]
        for r in range(0,args.R):
            mean = torch.matmul(torch.matmul(transformed_X_test, sampled_weights['a'][r,:,:]), sampled_weights['b'][r,:,:])
            pred_prob += (2 * np.pi) ** (-output_dim / 2) * torch.exp(
                -(1 / 2) * torch.norm(Y_test - mean, dim=1) ** 2)
        return -torch.log(pred_prob / args.R).mean()


def lastlayerIVI(model, args, X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    
    transformed_X_train = model.feature_map(X_train)
    transformed_X_valid = model.feature_map(X_valid)
    transformed_X_test = model.feature_map(X_test)

    transformed_train_loader = torch.utils.data.DataLoader(
        TensorDataset(Tensor(transformed_X_train), torch.as_tensor(Y_train, dtype=torch.long)),
        batch_size=args.batchsize, shuffle=True)
    transformed_valid_loader = torch.utils.data.DataLoader(
        TensorDataset(Tensor(transformed_X_valid), torch.as_tensor(Y_valid, dtype=torch.long)), 
        batch_size=args.batchsize, shuffle=True)

    return posterior_sample(args, transformed_train_loader, transformed_valid_loader, transformed_X_test, Y_test)
    
# def lastlayerlaplace():
    
    
def main():

    parser = argparse.ArgumentParser(description='last layer Bayesian')

    parser.add_argument('--taskid', type=int, default=1)

    # Data
    parser.add_argument('--input-dim', type=int, default=3)
    parser.add_argument('--output-dim', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='reducedrank_synthetic') #name should match last layer

    # Model
    parser.add_argument('--feature-map-hidden',type=int,default=5)
    parser.add_argument('--H', type=int, default=3, help='hidden units in final reduced regression layers')

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
    parser.add_argument('--MCs', type=int, default=50,
                        help='number of times to split into train-test')

    parser.add_argument('--R', type=int, default=1000,
                        help='number of MC draws from approximate posterior (default:200)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.epsilon_dim = args.H*(args.input_dim + args.output_dim)
    args.w_dim = args.H*(args.input_dim + args.output_dim)

    avg_lastlayerbayes_gen_err = np.array([])
    # avg_lastlayerlaplace_gen_err = np.array([])
    avg_map_gen_err = np.array([])
    n_range = np.array([100, 200, 500, 750, 1000, 1500])
    # n_range = np.array([100, 250])

    for n in n_range:

        map_gen_err = np.empty(args.MCs)
        lastlayerbayes_gen_err = np.empty(args.MCs)
        # lastlayerlaplace_gen_err = np.empty(args.MCs)
        
        args.n = n

        for mc in range(0, args.MCs):

            train_loader, valid_loader, test_loader, baseline, gen_err_baseline = get_data(args)

            X_train = train_loader.dataset[:][0]
            Y_train = train_loader.dataset[:][1]
            X_valid = valid_loader.dataset[:][0]
            Y_valid = valid_loader.dataset[:][1]
            X_test = test_loader.dataset[:][0]
            Y_test = test_loader.dataset[:][1]

            model = map_train(args, X_train, Y_train, X_test, Y_test, baseline)
            
            model.eval()
            map_gen_err[mc] = -torch.log((2*np.pi)**(-args.output_dim /2) * torch.exp(-(1/2) * torch.norm(Y_test-model(X_test), dim=1)**2)).mean()

            Bmap = list(model.parameters())[-1]
            Amap = list(model.parameters())[-2]
            params = (args.input_dim, args.output_dim, np.linalg.matrix_rank(torch.matmul(Bmap,Amap).detach().numpy()), args.H)
            trueRLCT = theoretical_RLCT('rr', params)
            print('true RLCT {}'.format(trueRLCT))

            lastlayerbayes_gen_err[mc] = lastlayerIVI(model, args, X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
            # lastlayerlaplace_gen_err[mc] = lastlayerlaplace()

            print('n = {}, mc {}, gen error (without entropy term): map {}, bayes last layer {}, baseline {}'
                  .format(n, mc, map_gen_err[mc], lastlayerbayes_gen_err[mc],gen_err_baseline))

            # true_rlct[mc] = get_true_rlct()
            # print('last layer reduced rank {}: mc {}: Bg {} true rlct {}'.format(H,mc, Bg[mc], true_rlct[mc]))

        print('average gen error (without entropy term): MAP {}, bayes {}, baseline {}'
              .format(map_gen_err.mean(),lastlayerbayes_gen_err.mean(), gen_err_baseline))
        # print('hat RLCT/n: {}'.format(rlct.mean() / n))
        # results.append({'H':H,'E_n Bg(n)': Bg.mean(), 'hat RLCT/n': rlct.mean()/ n})

        # avg_lastlayerlaplace_gen_err = np.append(avg_lastlayerlaplace_gen_err,lastlayerbayes_gen_err.mean())
        avg_lastlayerbayes_gen_err = np.append(avg_lastlayerbayes_gen_err, lastlayerbayes_gen_err.mean())
        avg_map_gen_err = np.append(avg_map_gen_err, map_gen_err.mean())

    print('avg bayes gen err {}'.format(avg_lastlayerbayes_gen_err))
    print('avg MAP gen err{}'.format(avg_map_gen_err))

    ols_model = OLS(avg_lastlayerbayes_gen_err, add_constant(1 / n_range)).fit()
    ols_intercept_estimate = ols_model.params[0]
    ols_slope_estimate = ols_model.params[1]
    print('estimated RLCT {}'.format(ols_intercept_estimate))
    #
    plt.scatter(1/n_range, avg_lastlayerbayes_gen_err, c='r', label='En G(n) for last layer Bayes predictive')
    plt.scatter(1/n_range, avg_map_gen_err, c='g', label='En G(n) for MAP')
    plt.plot(1 / n_range, ols_intercept_estimate + ols_slope_estimate / n_range, 'b-', label='ols')
    plt.xlabel('1/n')
    plt.title('slope {}, true RLCT {}'.format(ols_slope_estimate, trueRLCT))
    plt.legend()
    plt.savefig('taskid{}.png'.format(args.taskid))
    plt.show()


if __name__ == "__main__":
    main()