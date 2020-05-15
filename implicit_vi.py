from __future__ import print_function

import torch.optim as optim
import copy
import itertools

from RLCT_helper import *


class Discriminator(nn.Module):
    """
    input layer dim = w_dim, output layer dim = 1
    first layer Linear(w_dim, n_hidden_D) followed by ReLU
    num_hidden_layers_D of Linear(n_hidden_D, n_hidden_D) followed by ReLU
    final layer Linear(n_hidden_D, 1)
    """

    def __init__(self, w_dim, n_hidden_D, num_hidden_layers_D=2):
        super(Discriminator, self).__init__()

        self.enc_sizes = np.concatenate(
            ([w_dim], np.repeat(n_hidden_D, num_hidden_layers_D + 1), [1])).tolist()
        blocks = [[nn.Linear(in_f, out_f), nn.ReLU()]
                  for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        blocks = list(itertools.chain(*blocks))
        del blocks[-1]  # remove the last ReLu, don't need it in output layer

        self.net = nn.Sequential(*blocks)

    def forward(self, w):
        return self.net(w)


class Generator(nn.Module):
    """
    input layer dim = epsilon_dim, output layer dim = w_dim
    first layer Linear(epsilon_dim, n_hidden_G) followed by ReLU
    num_hidden_layers_G of Linear(n_hidden_G, n_hidden_G) followed by ReLU
    final layer Linear(n_hidden_G, w_dim)
    """

    def __init__(self, epsilon_dim, w_dim, n_hidden_G, num_hidden_layers_G=2):
        super(Generator, self).__init__()

        self.enc_sizes = np.concatenate(
            ([epsilon_dim], np.repeat(n_hidden_G, num_hidden_layers_G + 1), [w_dim])).tolist()
        blocks = [[nn.Linear(in_f, out_f), nn.ReLU()]
                  for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        blocks = list(itertools.chain(*blocks))
        del blocks[-1]  # remove the last ReLu, don't need it in output layer

        self.net = nn.Sequential(*blocks)

    def forward(self, epsilon):
        return self.net(epsilon)


# TODO: this needs to be put into the pyvarinf framework as Mingming has demonstrated in main_ivi and implicit_vi.py
def train_implicitVI(train_loader, valid_loader, args, mc, beta_index, saveimgpath):

    # instantiate generator and discriminator
    G_initial = Generator(args.epsilon_dim, args.w_dim, args.n_hidden_G, args.num_hidden_layers_G)  # G = Generator(args.epsilon_dim, w_dim).to(args.cuda)
    D_initial = Discriminator(args.w_dim, args.n_hidden_D, args.num_hidden_layers_D)  # D = Discriminator(w_dim).to(args.cuda)
    G = copy.deepcopy(G_initial)
    D = copy.deepcopy(D_initial)

    # optimizers
    opt_primal = optim.Adam(
        G.parameters(),
        lr=args.lr_primal)
    opt_dual = optim.Adam(
        D.parameters(),
        lr=args.lr_dual)
    # scheduler_G = ReduceLROnPlateau(opt_primal, mode='min', factor=0.1, patience=3, verbose=True)
    # early_stopping = EarlyStopping(patience=10, verbose=True)

    # pretrain discriminator
    for epoch in range(args.pretrainDepochs):
        w_sampled_from_prior = randn((args.batchsize, args.w_dim), args.cuda)
        eps = randn((args.batchsize, args.epsilon_dim), args.cuda)
        w_sampled_from_G = G(eps)
        loss_dual = torch.mean(-F.logsigmoid(D(w_sampled_from_G)) - F.logsigmoid(-D(w_sampled_from_prior)))

        loss_dual.backward()
        opt_dual.step()
        G.zero_grad()
        D.zero_grad()

    train_loss_epoch, valid_loss_epoch, train_reconstr_err_epoch, valid_reconstr_err_epoch, D_err_epoch = [], [], [], [], []
    reconstr_err_minibatch, D_err_minibatch, primal_loss_minibatch = [], [], []

    # TODO: put back logging
    itr = 0 # iteration counter
    # train discriminator and generator together
    for epoch in range(args.epochs):

        D.train()
        G.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            # opt discriminator more than generator
            for discriminator_epoch in range(args.trainDepochs):
                w_sampled_from_prior = randn((args.epsilon_mc, args.w_dim),args.cuda)
                eps = randn((args.epsilon_mc, args.epsilon_dim), args.cuda)
                loss_dual = torch.mean(-F.logsigmoid(D(G(eps))) - F.logsigmoid(-D(w_sampled_from_prior)))
                loss_dual.backward()
                opt_dual.step()
                G.zero_grad()
                D.zero_grad()

            data, target = load_minibatch(args, data, target)

            # opt generator
            eps = randn((args.epsilon_mc, args.epsilon_dim), args.cuda)
            sampled_weights = G(eps) # [args.epsilon_mc, w_dim]
            batch_ELBO_reconstr = 0

            for i in range(args.epsilon_mc):  # loop over rows of sampled_weights corresponding to different epsilons
                current_w = sampled_weights[i, :].unsqueeze(dim=0)
                param_dict = weights_to_dict(args, current_w)[0]
                batch_ELBO_reconstr += calculate_nllsum_paramdict(args, target, data, param_dict)

            reconstr_err_component = batch_ELBO_reconstr / (args.epsilon_mc * len(target))
            discriminator_err_component = torch.mean(D(sampled_weights)) / (args.betas[beta_index] * args.n)
            loss_primal = reconstr_err_component + discriminator_err_component
            loss_primal.backward(retain_graph=True)
            itr += 1
            opt_primal.step()
            G.zero_grad()
            D.zero_grad()

            reconstr_err_minibatch.append(reconstr_err_component.item())
            D_err_minibatch.append(discriminator_err_component.item())
            primal_loss_minibatch.append(loss_primal.item())

            # if itr % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss primal: {:.6f}\tLoss dual: {:.6f}'.format(
            #             epoch, batch_idx * len(data), args.n, 100. * batch_idx / len(train_loader), loss_primal.data.item(), loss_dual.data.item()))

        if epoch % args.log_interval == 0:
            with torch.no_grad():
                nllw_array_train = approxinf_nll_implicit(train_loader, G, args)
                nllw_array_valid = approxinf_nll_implicit(valid_loader, G, args)

            print('Train Epoch: {} '
                  '\tLoss primal: {:.6f} '
                  '\tLoss dual: {:.6f} '
                  '\t L_n(w) training: {:.6f} '
                  '\t L_n(w) valid: {:.6f}'
                  .format(epoch,
                          loss_primal.data.item(),
                          loss_dual.data.item(),
                          nllw_array_train.mean()/len(train_loader.dataset),
                          nllw_array_valid.mean()/len(valid_loader.dataset)))

        # with torch.no_grad(): #to save memory, no intermediate activations used for gradient calculation is stored.
        #     D.eval()
        #     G.eval()
        #     #valid loss is calculated for all monte carlo as it is used for scheduler_G
        #     valid_loss_minibatch = []
        #     for valid_batch_id, (valid_data, valid_target) in enumerate(valid_loader):
        #         valid_data, valid_target = load_minibatch(args, valid_data, valid_target)
        #         if args.dataset == 'tanh_synthetic':
        #            valid_output = torch.matmul(torch.tanh(torch.matmul(valid_data, torch.transpose(a_params, 0, 1))), torch.transpose(b_params, 0, 1))
        #            valid_loss_minibatch.append(args.loss_criterion(valid_output, valid_target).item() * 0.5)
        #         elif args.dataset =='logistic_synthetic':
        #             if args.bias:
        #                 output = torch.mm(valid_data, A.reshape(args.w_dim - 1, 1)) + b
        #             else:
        #                 output = torch.mm(valid_data, A.reshape(args.w_dim, 1))
        #             output_cat_zero = torch.cat((output, torch.zeros(valid_data.shape[0], 1)), 1)
        #             valid_output = F.log_softmax(output_cat_zero, dim=1)
        #             valid_loss_minibatch.append(args.loss_criterion(torch.sigmoid(output), valid_target).item())
        #         elif args.dataset == 'reducedrank_synthetic':
        #            valid_output = torch.matmul(torch.matmul(valid_data, torch.transpose(a_params, 0, 1)), torch.transpose(b_params, 0, 1))
        #            valid_loss_minibatch.append(args.loss_criterion(valid_output, valid_target).item()*0.5)
        #     valid_loss_one = np.average(valid_loss_minibatch) + torch.mean(D(w_sampled_from_G)) / (args.betas[beta_index]  * args.n)
        #     scheduler_G.step(valid_loss_one)
        #     valid_loss_epoch.append(valid_loss_one)
        #     valid_reconstr_err_epoch.append(np.average(valid_loss_minibatch))
        #     D_err_epoch.append(torch.mean(D(w_sampled_from_G)) / (args.betas[beta_index]  * args.n))
        #
        #     train_loss_minibatch2 = []
        #     for train_batch_id, (train_data, train_target) in enumerate(train_loader):
        #         train_data, train_target = load_minibatch(args, train_data, train_target)
        #         if args.dataset == 'tanh_synthetic':
        #             train_output = torch.matmul(torch.tanh(torch.matmul(train_data, torch.transpose(a_params, 0, 1))), torch.transpose(b_params, 0, 1))
        #             train_loss_minibatch2.append(args.loss_criterion(train_output, train_target).item()*0.5)
        #         elif args.dataset == 'logistic_synthetic':
        #             if args.bias:
        #                 output = torch.mm(train_data, A.reshape(args.w_dim - 1, 1)) + b
        #             else:
        #                 output = torch.mm(train_data, A.reshape(args.w_dim, 1))
        #
        #             output_cat_zero = torch.cat((output, torch.zeros(train_data.shape[0], 1)), 1)
        #             train_output = F.log_softmax(output_cat_zero, dim=1)
        #             train_loss_minibatch2.append(args.loss_criterion(torch.sigmoid(output), train_target).item())
        #         elif args.dataset == 'reducedrank_synthetic':
        #             train_output = torch.matmul(torch.matmul(train_data, torch.transpose(a_params, 0, 1)), torch.transpose(b_params, 0, 1))
        #             train_loss_minibatch2.append(args.loss_criterion(train_output, train_target).item()*0.5)
        #     train_loss_epoch.append(np.average(train_loss_minibatch2) + torch.mean(D(w_sampled_from_G)) / (args.betas[beta_index]  * args.n))
        #     train_reconstr_err_epoch.append(np.average(train_loss_minibatch2))

        # early_stopping(valid_loss_one, G)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    # plt.figure(figsize=(10, 7))
    # plt.plot(list(range(0, len(train_loss_epoch))), train_loss_epoch,
    #          list(range(0, len(valid_loss_epoch))), valid_loss_epoch,
    #          list(range(0, len(train_reconstr_err_epoch))), train_reconstr_err_epoch,
    #          list(range(0, len(valid_reconstr_err_epoch))), valid_reconstr_err_epoch,
    #          list(range(0, len(D_err_epoch))), D_err_epoch)
    # plt.legend(('primal loss (train)', 'primal loss (validation)', 'reconstr err component (train)', 'reconstr err component (valid)', 'discriminator err component'), loc='center right', fontsize=16)
    # plt.xlabel('epoch', fontsize=16)
    # plt.title('beta = {}'.format(args.betas[beta_index]), fontsize=18)
    # plt.close()
    #
    # plt.figure(figsize=(10, 7))
    # plt.plot(list(range(0, len(reconstr_err_minibatch)))[20:], reconstr_err_minibatch[20:],
    #          list(range(0, len(D_err_minibatch)))[20:], D_err_minibatch[20:],
    #          list(range(0, len(primal_loss_minibatch)))[20:], primal_loss_minibatch[20:]
    # )
    #
    # plt.legend(('reconstr err component', 'discriminator err component','primal loss'), loc='upper right', fontsize=16)
    # plt.xlabel('epochs*batches (minibatches)', fontsize=16)
    # plt.title('training_set, beta = {}'.format(args.betas[beta_index]), fontsize=18)
    # plt.savefig('./{}/reconsterr_derr_minibatch_betaind{}.png'.format(saveimgpath, beta_index))
    # plt.close()

    return G


def sample_IVI(args, G, num_draws):
    # given trained generator G, returns one random draw w^* from G

    G.eval()
    with torch.no_grad():

        eps = randn((num_draws, args.epsilon_dim), args.cuda)
        sampled_weights = G(eps)

        list_of_param_dicts = weights_to_dict(args, sampled_weights)

    return list_of_param_dicts


# Draw w^* from generator G and evaluate nL_n(w^*) on train_loader, perform args.R times and return array
def approxinf_nll_implicit(train_loader, G, args):

    param_dicts = sample_IVI(args, G, num_draws=args.R)

    nllw_array = np.array([])
    for param_dict in param_dicts:
        nllw_array = np.append(nllw_array, calculate_nllsum_paramdict(args, train_loader.dataset[:][1], train_loader.dataset[:][0], param_dict))

    return nllw_array

