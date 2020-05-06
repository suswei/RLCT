from __future__ import print_function

import torch.optim as optim
import copy

import pyvarinf
from RLCT_helper import *


def train_explicitVI(train_loader, valid_loader, args, mc, beta_index, verbose, saveimgpath):

    # retrieve model
    model, _ = retrieve_model(args)

    # variationalize model
    var_model_initial = pyvarinf.Variationalize(model, zero_mean=False)  # TODO: does zero_mean have a meaningful impact?

    # setting up prior parameters
    prior_parameters = {}
    if args.prior != 'gaussian':
        prior_parameters['n_mc_samples'] = 10
    if args.prior == 'mixtgauss':
        prior_parameters['sigma_1'] = 0.02
        prior_parameters['sigma_2'] = 0.2
        prior_parameters['pi'] = 0.5
    if args.prior == 'conjugate':
        prior_parameters['mu_0'] = 0.
        prior_parameters['kappa_0'] = 3.
        prior_parameters['alpha_0'] = .5
        prior_parameters['beta_0'] = .5
    if args.prior == 'conjugate_known_mean':
        prior_parameters['alpha_0'] = .5
        prior_parameters['beta_0'] = .5
        prior_parameters['mean'] = 0.

    var_model = copy.deepcopy(var_model_initial)
    var_model.set_prior(args.prior, **prior_parameters)
    if args.cuda:
        var_model.cuda()
    optimizer = optim.Adam(var_model.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    # early_stopping = EarlyStopping(patience=10, verbose=True)

    # TODO: put back logging
    train_loss_epoch, valid_loss_epoch, train_reconstr_err_epoch, valid_reconstr_err_epoch, loss_prior_epoch = [], [], [], [], []
    reconstr_err_minibatch, loss_prior_minibatch, train_loss_minibatch = [], [], []
    itr = 0 # iteration counter
    # train var_model
    for epoch in range(1, args.epochs + 1):

        var_model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = load_minibatch(args, data, target)
            optimizer.zero_grad()
            output = var_model(data)
            loss_prior = var_model.prior_loss() / (args.betas[beta_index] * args.n)

            if args.dataset == 'logistic_synthetic':
                reconstr_err = args.loss_criterion(output, target)
            elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
                reconstr_err = args.loss_criterion(output, target) * 0.5

            loss = reconstr_err / len(target) + loss_prior  # this is the ELBO
            loss.backward()
            itr += 1
            optimizer.step()

            reconstr_err_minibatch.append(reconstr_err.item())
            loss_prior_minibatch.append(loss_prior.item())
            train_loss_minibatch.append(loss.item())

            # if itr % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss error: {:.6f}\tLoss weights: {:.6f}'.format(
            #             epoch, batch_idx * len(data), args.n,
            #                    100. * batch_idx / len(train_loader), loss.data.item(),
            #                    reconstr_err.data.item() / len(target), loss_prior.data.item()))

        if epoch % args.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}\tLoss error: {:.6f}\tLoss weights: {:.6f}'.format(
                epoch, loss.data.item(), reconstr_err.data.item()/len(target), loss_prior.data.item()))

    #     with torch.no_grad():  # to save memory, no intermediate activations used for gradient calculation is stored.
    #         var_model.eval()
    #         # valid loss is calculated for all monte carlo as it is used for scheduler_G
    #         valid_loss_minibatch = []
    #         for valid_batch_id, (valid_data, valid_target) in enumerate(valid_loader):
    #             valid_data, valid_target = load_minibatch(args, valid_data, valid_target)
    #             valid_output = var_model(valid_data)
    #             if args.dataset == 'logistic_synthetic':
    #                 valid_loss_minibatch.append(args.loss_criterion(valid_output, valid_target).item())
    #             elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
    #                 valid_loss_minibatch.append(args.loss_criterion(valid_output, valid_target).item()*0.5)
    #         valid_loss_one = np.average(valid_loss_minibatch) + var_model.prior_loss() / (args.betas[beta_index] * args.n)
    #         scheduler.step(valid_loss_one)
    #         valid_loss_epoch.append(valid_loss_one)
    #         valid_reconstr_err_epoch.append(np.average(valid_loss_minibatch))
    #         loss_prior_epoch.append(var_model.prior_loss() / (args.betas[beta_index] * args.n))
    #
    #         train_loss_minibatch2 = []
    #         for train_batch_id, (train_data, train_target) in enumerate(train_loader):
    #             train_data, train_target = load_minibatch(args, train_data, train_target)
    #             train_output = var_model(train_data)
    #             if args.dataset == 'logistic_synthetic':
    #                 train_loss_minibatch2.append(args.loss_criterion(train_output, train_target).item())
    #             elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
    #                 train_loss_minibatch2.append(args.loss_criterion(train_output, train_target).item()*0.5)
    #         train_loss_epoch.append(np.average(train_loss_minibatch2)+var_model.prior_loss() / (args.betas[beta_index] * args.n))
    #         train_reconstr_err_epoch.append(np.average(train_loss_minibatch2))
    #
    #     # early_stopping(valid_loss_one, var_model)
    #     # if early_stopping.early_stop:
    #     #     print("Early stopping")
    #     #     break
    #
    # plt.figure(figsize=(10, 7))
    # plt.plot(list(range(0, len(train_loss_epoch))), train_loss_epoch,
    #          list(range(0, len(valid_loss_epoch))), valid_loss_epoch,
    #          list(range(0, len(train_reconstr_err_epoch))), train_reconstr_err_epoch,
    #          list(range(0, len(valid_reconstr_err_epoch))), valid_reconstr_err_epoch,
    #          list(range(0, len(loss_prior_epoch))), loss_prior_epoch)
    # plt.legend(('loss (train)', 'loss (validation)', 'reconstr err component (train)',
    #             'reconstr err component (valid)', 'loss prior component'), loc='center right', fontsize=16)
    # plt.xlabel('epoch', fontsize=16)
    # plt.title('beta = {}'.format(args.betas[beta_index]), fontsize=18)
    # plt.savefig('./{}/primal_loss_betaind{}.png'.format(saveimgpath, beta_index))
    # plt.close()
    #
    # plt.figure(figsize=(10, 7))
    # plt.plot(list(range(0, len(reconstr_err_minibatch)))[20:], reconstr_err_minibatch[20:],
    #          list(range(0, len(loss_prior_minibatch)))[20:], loss_prior_minibatch[20:],
    #          list(range(0, len(train_loss_minibatch)))[20:], train_loss_minibatch[20:])
    #
    # plt.legend(('reconstr err component', 'loss prior component', 'loss'), loc='upper right',fontsize=16)
    # plt.xlabel('epochs*batches (minibatches)', fontsize=16)
    # plt.title('training_set, beta = {}'.format(args.betas[beta_index]), fontsize=18)
    # plt.savefig('./{}/reconsterr_derr_minibatch_betaind{}.png'.format(saveimgpath, beta_index))
    # plt.close()

    return var_model


def sample_EVI(var_model, args):
# given trained var_model, draw many weights

    sampled_weight = torch.empty((0, args.w_dim))
    for draw_index in range(100):

        if args.dataset == 'logistic_synthetic':

            temp = var_model.dico['linear']['weight']
            weight = temp.mean + (1 + temp.rho.exp()).log() * torch.randn(temp.mean.shape) # drawing epsilon from Gaussian
            sampled_weight = torch.cat((sampled_weight, weight))

        elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:

            temp = var_model.dico['fc1']['weight']
            fc1_weight = temp.mean + (1 + temp.rho.exp()).log() * torch.randn(temp.mean.shape)
            temp = var_model.dico['fc2']['weight']
            fc2_weight = temp.mean + (1 + temp.rho.exp()).log() * torch.randn(temp.mean.shape)

            weight = torch.cat((fc1_weight,fc2_weight),1)
            sampled_weight = torch.cat((sampled_weight, weight))
            #
            # layer1_weight_mean_rho_eps = list(list(var_model.dico.values())[0].values())[0]
            # layer2_weight_mean_rho_eps = list(list(var_model.dico.values())[1].values())[0]
            #
            # layer1_weight = layer1_weight_mean_rho_eps.mean + (
            #             1 + layer1_weight_mean_rho_eps.rho.exp()).log() * layer1_weight_mean_rho_eps.eps  # H * input_dim
            # layer2_weight = layer2_weight_mean_rho_eps.mean + (
            #             1 + layer2_weight_mean_rho_eps.rho.exp()).log() * layer2_weight_mean_rho_eps.eps  # output_dim * H
            #
            # one_weight = torch.cat((layer1_weight.reshape(1, (layer1_weight.shape[0] * layer1_weight.shape[1])),
            #                         layer2_weight.reshape(1, (layer2_weight.shape[0] * layer2_weight.shape[1]))), 1)
            # sampled_weight = torch.cat((sampled_weight, one_weight), 0)

    return sampled_weight


# TODO: approxinf_nll_explicit should perhaps follow approxinf_nll_implicit format
# Draw w^* by calling sample.draw() and evaluate nL_n(w^*) on train_loader, perform args.R times and return array
def approxinf_nll_explicit(train_loader, var_model, args):

    wholex = train_loader.dataset[:][0]
    wholey = train_loader.dataset[:][1]
    sample = pyvarinf.Sample(var_model=var_model)

    nllw_array = np.array([])
    for r in range(0, args.R):
        sample.draw()
        output = sample(wholex)

        if args.dataset == 'logistic_synthetic':
            nllw_array = np.append(nllw_array, args.loss_criterion(output, wholey).detach().numpy())

        elif args.dataset in ['tanh_synthetic', 'reducedrank_synthetic']:
            nllw_array = np.append(nllw_array, args.n * args.output_dim * 0.5 * np.log(2 * np.pi) + 0.5 * args.loss_criterion(output, wholey).detach().numpy())
        else:
            print('misspelling in dataset name!')

    return nllw_array