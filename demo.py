from __future__ import print_function
from main import *
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


# Let's set some common arguments
class Args:
    syntheticsamplesize = 500
    batchsize = 100
    w_dim = 2
    dpower = None
    posterior_viz = True

    epochs = 200
    prior = 'gaussian'
    pretrainDepochs = 100
    trainDepochs = 50
    n_hidden_D = 128
    num_hidden_layers_D = 1
    n_hidden_G = 128
    num_hidden_layers_G = 1

    lr_primal = 1e-3
    lr_dual = 1e-3
    lr = 1e-2

    elasticnet_alpha = 0.5

    beta_auto_liberal = False
    beta_auto_conservative = False
    beta_auto_oracle = False
    betasbegin = 0.1
    betasend = 1.9
    betalogscale = True
    numbetas = 3

    R = 200

    cuda = False

    log_interval = 100
args = Args()


def main(args):

    # draw new training-testing split
    train_loader, valid_loader, test_loader = get_dataset_by_id(args, kwargs)

    # get a grid of inverse temperatures [beta_1/log n, \ldots, beta_k/log n]
    set_betas(args)

    mc = 1
    saveimgpath = None
    nll_betas_explicit = np.empty(0)
    nll_betas_implicit = np.empty(0)

    for beta_index in range(args.betas.shape[0]):

        # train explicit variational inference
        var_model = train_explicitVI(train_loader, valid_loader, args, mc, beta_index, True, saveimgpath)
        nllw_array_explicit = approxinf_nll_explicit(train_loader, var_model, args)
        # record E nL_n(w)
        nll_betas_explicit = np.append(nll_betas_explicit, nllw_array_explicit.mean())

        # visualize EVI
        args.VItype = 'explicit'
        sampled_weights = sample_EVI(var_model, args)
        posterior_viz(train_loader, sampled_weights, args, beta_index, saveimgpath)

        # train implicit variational inference
        args.epsilon_dim = args.w_dim
        args.epsilon_mc = args.batchsize
        args.VItype = 'implicit'
        G = train_implicitVI(train_loader, valid_loader, args, mc, beta_index, saveimgpath)
        nllw_array_implicit = approxinf_nll_implicit(train_loader, G, args)
        nll_betas_implicit = np.append(nll_betas_implicit, nllw_array_implicit.mean())

        # visualize IVI
        with torch.no_grad():
            eps = torch.randn(100, args.epsilon_dim)
            sampled_weights = G(eps)
            posterior_viz(train_loader, sampled_weights, args, beta_index, saveimgpath)


    # should observe a straight line below
    lsfit_lambda(nll_betas_explicit, args, saveimgpath)
    lsfit_lambda(nll_betas_implicit, args, saveimgpath)


## logistic regression 2D

args.dataset = 'logistic_synthetic'
args.network = 'logistic'
args.bias = False
args.input_dim = args.w_dim
args.output_dim = 1

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# true data is generated from y = bernoulli(p), where p = 1/(1+e^-(w^T x + b)
# set true parameters to w = [0.5, 1] and b = 0.0

# Let's generate some data according to this model
args.w_0 = torch.Tensor([[0.5], [1]])
args.b = torch.tensor([0.0])
X = torch.randn(2 * args.syntheticsamplesize, args.input_dim)
affine = torch.mm(X, args.w_0) + args.b
m = torch.distributions.bernoulli.Bernoulli(torch.sigmoid(affine))
y = m.sample()

# note that this is a regular model, in particular K(w) = 0 is a singleton consisting of w_0

# Let's first visualize the data
plt.plot(affine.squeeze(dim=1).detach().numpy(), y.detach().numpy(), '.g')
plt.plot(affine.squeeze(dim=1).detach().numpy(), torch.sigmoid(affine).detach().numpy(), '.r')
plt.title('synthetic logistic regression data: w^T x + b versus probabilities and Bernoulli(p)')
plt.show()

main(args)

## tanh regression 2D

args.dataset = 'tanh_synthetic'
args.network = 'tanh'
args.bias = False
args.H = 1
args.a_params = torch.zeros([1, args.H], dtype=torch.float32)
args.b_params = torch.zeros([args.H, 1], dtype=torch.float32)
args.input_dim = 1
args.output_dim = 1

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# true data is generated from y = normal(f(x,a,b),1), model is f(x,a,b) = a \tanh bx
# set true parameters to [a,b]=[0,0]

# Let's generate some data according to this model
m = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
X = m.sample(torch.Size([2 * args.syntheticsamplesize]))
# w = {(a_m,b_m)}_{m=1}^p, p(y|x,w) = N(0,f(x,w)) where f(x,w) = \sum_{m=1}^p a_m tanh(b_m x)
mean = torch.matmul(torch.tanh(torch.matmul(X, args.a_params)), args.b_params)
y_rv = Normal(mean, 1)
y = y_rv.sample()

# Let's first visualize the data
plt.plot(X.detach().numpy(), y.detach().numpy(), '.g')
plt.title('synthetic tanh regression data: x versus y')
plt.show()

# note that this is a singular model. \{a,b: K(a,b) = 0\} = \{a,b \in \mathbb R: ab = 0}


# Now let's train explicit and implicit VI
main(args)


## reduced rank regression 2D

args.dataset = 'reducedrank_synthetic'
args.network = 'reducedrank'
args.bias = False
args.a_params = torch.Tensor([1.0]).reshape(1, 1)
args.b_params = torch.Tensor([1.0]).reshape(1, 1)
args.input_dim = 1
args.output_dim = 1
args.H = 1

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# true data is generated from y = normal(f(x,a,b),1), model is f(x,a,b) = a bx
# set true parameters to [a,b]=[1,1]

# Let's generate some data according to this model
m = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(
    args.input_dim))  # the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
X = m.sample(torch.Size([2 * args.syntheticsamplesize]))
mean = torch.matmul(torch.matmul(X, args.a_params), args.b_params)
y_rv = MultivariateNormal(mean, torch.eye(args.output_dim))
y = y_rv.sample()

# Let's first visualize the data
plt.plot(X.detach().numpy(), y.detach().numpy(), '.g')
plt.title('synthetic tanh regression data: x versus y')
plt.show()

# note that this is a singular model. \{a,b: K(a,b) = 0\} = \{a,b \in \mathbb R: ab = 1}

# Now let's train explicit and implicit VI
main(args)



