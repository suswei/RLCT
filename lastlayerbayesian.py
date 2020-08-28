# wiseodd/last_layer_laplace

import matplotlib
from torch.distributions.multivariate_normal import MultivariateNormal

import seaborn as sns; sns.set_style('white')

from torch.utils.data import TensorDataset
from torch import Tensor

from main import *

plt = matplotlib.pyplot

# dataset

def get_data(n, args):

    # The splitting ratio of training set, validation set, testing set is 0.7:0.15:0.15
    train_size = n
    valid_size = int(n * 0.5)
    test_size = 10000

    a = Normal(0.0, 1.0)
    a_params = 0.2 * a.sample((args.input_dim, args.H))
    b = Normal(0.0, 1.0)
    b_params = 0.2 * b.sample((args.H, args.output_dim))
    m = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(args.input_dim))
    X = 3.0 * m.sample(torch.Size([train_size+valid_size+test_size]))

    true_mean = torch.matmul(torch.matmul(X, a_params), b_params)
    y_rv = MultivariateNormal(torch.zeros(args.output_dim), torch.eye(args.output_dim))
    y = true_mean + 0.1 * y_rv.sample(torch.Size([train_size+valid_size+test_size]))

    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(X, y),
                                                                               [train_size, valid_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True)


    return train_loader, valid_loader, test_loader


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
            # nn.BatchNorm1d(input_dim),
            # nn.ReLU(),
        )

        self.clf = nn.Sequential(
            nn.Linear(input_dim, rr_H, bias=False),
            nn.Linear(rr_H, output_dim, bias=False)
        )

    def forward(self, x):
        x = self.feature_map(x)
        return self.clf(x)


# train module
def map_train(args, X_train, Y_train, X_test, Y_test):

    n, input_dim = X_train.shape
    n, output_dim = Y_train.shape

    model = Model(input_dim, output_dim, args.feature_map_hidden, args.H)
    opt = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)

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
            print('loss: {:.3f}, test loss: {:.3f}'.format(l.item(),test_loss.item()))

    return model


# log p(y_i |x_i, D_n) on test set
def compute_predictive_dist(X_test, Y_test, last_layer_samples):

    pred_logprob = 0
    for param_dict in last_layer_samples:

        mean = torch.matmul(torch.matmul(X_test, param_dict['a']), param_dict['b'])
        pred_logprob -= torch.norm(Y_test-mean, dim=1)**2 / 2

    return pred_logprob.mean()/len(last_layer_samples)


def posterior_sample(args, train_loader, test_loader):

    mc = 1
    beta_index = 0
    args.betas = [1.0]
    saveimgpath = None

    G = train_implicitVI(train_loader, test_loader, args, mc, beta_index, saveimgpath)

    eps = torch.randn(args.R, args.epsilon_dim)
    sampled_weights = G(eps)
    list_of_param_dicts = weights_to_dict(args, sampled_weights)

    return list_of_param_dicts


def main():

    parser = argparse.ArgumentParser(description='last layer Bayesian')

    # Data
    parser.add_argument('--input-dim', type=int, default=20)
    parser.add_argument('--output-dim', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='reducedrank_synthetic')

    # Model
    parser.add_argument('--feature-map-hidden',type=int,default=10)
    parser.add_argument('--H', type=int, default=5, help='hidden units in final reduced regression layers')

    # Approximate Bayesian inference

    parser.add_argument('--batchsize', type=int, default=20, help='used in IVI')

    parser.add_argument('--epsilon_mc', type=int, default=100, help='used in IVI')

    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--pretrainDepochs', type=int, default=100,
                        help='number of epochs to pretrain discriminator')

    parser.add_argument('--trainDepochs', type=int, default=50,
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

    parser.add_argument('--lr_dual', type=float, default=0.005, metavar='LR',
                        help='dual learning rate (default: 0.01)')

    # averaging
    parser.add_argument('--MCs', type=int, default=5,
                        help='number of times to split into train-test')

    parser.add_argument('--R', type=int, default=200,
                        help='number of MC draws from approximate posterior (default:200)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.epsilon_dim = args.H*(args.input_dim + args.output_dim)
    args.w_dim = args.H*(args.input_dim + args.output_dim)

    avg_bayes_gen_err = np.array([])
    avg_map_gen_err = np.array([])
    n_range = np.array([100, 250, 500, 1000])
    # n_range = np.array([100, 250])

    for n in n_range:

        map_gen_err = np.empty(args.MCs)
        bayes_gen_err = np.empty(args.MCs)

        args.n = n

        for mc in range(0, args.MCs):

            train_loader, valid_loader, test_loader = get_data(n, args)

            X_train = train_loader.dataset[:][0]
            Y_train = train_loader.dataset[:][1]
            X_valid = valid_loader.dataset[:][0]
            Y_valid = valid_loader.dataset[:][1]
            X_test = test_loader.dataset[:][0]
            Y_test = test_loader.dataset[:][1]

            model = map_train(args, X_train, Y_train, X_test, Y_test)
            map_logprob_test = -torch.norm(Y_test - model(X_test), dim=1) ** 2 / 2
            map_gen_err[mc] = -map_logprob_test.mean()

            Bmap = list(model.parameters())[-1]
            Amap = list(model.parameters())[-2]
            params = (args.input_dim, args.output_dim, np.linalg.matrix_rank(torch.matmul(Amap,Bmap).detach().numpy()), args.H)
            trueRLCT = theoretical_RLCT('rr', params)
            print('true RLCT {}'.format(trueRLCT))

            transformed_X_train = model.feature_map(X_train)
            transformed_X_test = model.feature_map(X_test)

            temp_train = TensorDataset(Tensor(transformed_X_train), torch.as_tensor(Y_train, dtype=torch.long))
            new_trainloader = torch.utils.data.DataLoader(temp_train, batch_size=args.batchsize, shuffle=True)

            temp_test = TensorDataset(Tensor(transformed_X_test), torch.as_tensor(Y_test, dtype=torch.long))
            new_testloader = torch.utils.data.DataLoader(temp_test, batch_size=args.batchsize, shuffle=True)

            last_layer_samples = posterior_sample(args, new_trainloader, new_testloader)
            bayes_logprob_test = compute_predictive_dist(transformed_X_test, Y_test, last_layer_samples)
            bayes_gen_err[mc] = -bayes_logprob_test

            print('gen error without entropy term: map {}, bayes last layer {}'.format(map_gen_err[mc], bayes_gen_err[mc]))

            # true_rlct[mc] = get_true_rlct()
            # print('last layer reduced rank {}: mc {}: Bg {} true rlct {}'.format(H,mc, Bg[mc], true_rlct[mc]))

        print('average generalisation error (without entropy term): map {}, bayes {}'.format(map_gen_err.mean(),bayes_gen_err.mean()))
        # print('hat RLCT/n: {}'.format(rlct.mean() / n))
        # results.append({'H':H,'E_n Bg(n)': Bg.mean(), 'hat RLCT/n': rlct.mean()/ n})

        avg_bayes_gen_err = np.append(avg_bayes_gen_err, bayes_gen_err.mean())
        avg_map_gen_err = np.append(avg_map_gen_err, map_gen_err.mean())

    print('avg bayes gen err {}'.format(avg_bayes_gen_err))
    print('avg_map_gen_err{}'.format(avg_map_gen_err))

    # ols_model = OLS(avg_bayes_gen_err, add_constant(1 / n_range)).fit()
    # ols_intercept_estimate = ols_model.params[0]
    # ols_slope_estimate = ols_model.params[1]
    #
    # plt.scatter(n_range, avg_bayes_gen_err, c='r', label='En G(n) for last layer Bayes predictive')
    # plt.scatter(n_range, avg_map_gen_err, c='g', label='En G(n) for map')
    # plt.plot(1 / n_range, ols_intercept_estimate + ols_slope_estimate * 1 / n_range, 'b-', label='ols')
    # plt.title('slope {}, true RLCT {}'.format(ols_slope_estimate, trueRLCT))
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()