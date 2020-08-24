# The NUTS code here is modified from https://adamhaber.github.io/post/nuts/ with thanks!
import argparse
import os
import pickle
import math
import time
from datetime import datetime
from functools import partial
import collections

import numpy as np

import multiprocessing
from multiprocessing import Process, Manager

TraceReturns = collections.namedtuple('TraceReturns', ['target_log_prob',
                                                       'leapfrogs_taken',
                                                       'has_divergence',
                                                       'energy',
                                                       'log_accept_ratio',
                                                        'is_accepted',
                                                          'step_size'])

# For running chains in parallel
def run_chains(beta, beta_index, dataset_index, samples, timers, datasets, args):
    import tensorflow as tf
    import mcmc_symmetry
    
    # In TF we can run multiple processes using the GPU in parallel
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    #
    # However note that it is _not_ true that the TF processes actually
    # respect args.gpu_memory as a hard limit. E.g. this can be
    # set to 128 but the processes (viewed via nvidia-smi) will each consume
    # say 443Mb.
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate a fixed amount of memory
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.gpu_memory)])
            #tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    
    import tensorflow_probability as tfp

    from tensorflow_probability import distributions as tfd
    from tensorflow_probability import bijectors as tfb

    # https://github.com/tensorflow/probability/blob/7cf006d6390d1d0a6fe541e5c49196a58a22b875/tensorflow_probability/python/mcmc/nuts.py#L86

    def trace_fn_nuts(_, pkr):
        return TraceReturns(
            pkr.inner_results.inner_results.target_log_prob,
            pkr.inner_results.inner_results.leapfrogs_taken,
            pkr.inner_results.inner_results.has_divergence,
            pkr.inner_results.inner_results.energy,
            pkr.inner_results.inner_results.log_accept_ratio,
            pkr.inner_results.inner_results.is_accepted,
            pkr.inner_results.inner_results.step_size,
        )

    def run_nuts(
        target_log_prob_fn,
        inits,
        bijectors_list=None,
        num_burnin_steps=1000,
        num_results=1000,
        num_chains=1):

        step_size = np.random.rand(num_chains, 1)*.5 + 1.

        if not isinstance(inits, list):
            inits = [inits]

        if bijectors_list is None:
            bijectors_list = [tfb.Identity()]*len(inits)

        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.NoUTurnSampler(
                    target_log_prob_fn,
                    step_size=[step_size]*len(inits)
                ),
                bijector=bijectors_list
            ),
            target_accept_prob=args.target_accept_prob,
            num_adaptation_steps=int(0.8*num_burnin_steps),
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                  inner_results=pkr.inner_results._replace(step_size=new_step_size)
              ),
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
        )

        res = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=inits,
            kernel=kernel,
            trace_fn=trace_fn_nuts
        )
        return res

    run_nuts_opt = tf.function(run_nuts,autograph=False,experimental_compile=True)

    # NOTE: to get XLA to work I needed to make a symlink
    # https://github.com/google/jax/issues/989

    true_dist_state = mcmc_symmetry.true_distribution(args.num_hidden_true,args.num_hidden)

    def swish(X):
        return 1/args.swish_beta * tf.nn.swish( args.swish_beta * X )
        
    if( args.nonlinearity == "relu" ):
        nl = tf.nn.relu
    else:
        nl = swish
            
    true_network = mcmc_symmetry.build_network(true_dist_state[::2],
                                 true_dist_state[1::2], nl)

    X_train, y_train = mcmc_symmetry.generate_data(args.num_data, 
                                                    true_network, 
                                                    args.x_max)

    center = true_dist_state

    NUM_PARAMS = 0
    for s in true_dist_state:
      print("State shape", s.shape)
      NUM_PARAMS += s.shape.num_elements()
    print("Total params", NUM_PARAMS)

    weight_prior = tfd.Normal(0.0, args.prior_sd)

    logp = partial(mcmc_symmetry.joint_log_prob_fn, center, weight_prior,
                        X_train, y_train, beta, nl)

    center_nuts = [tf.expand_dims(x,axis=0) for x in center]

    pre_time = time.time()
  
    fine_chain, fine_trace = run_nuts_opt(
        logp,
        center_nuts,
        num_burnin_steps=args.num_warmup,
        num_results=args.num_samples)

    post_time = time.time()
    delta_time = post_time - pre_time
    
    samples[(beta_index,dataset_index)] = [fine_chain, fine_trace]
    datasets[(beta_index,dataset_index)] = [X_train, y_train]
    timers[(beta_index,dataset_index)] = delta_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RLCT_HMC_symmetric")
    parser.add_argument("--experiment-id", nargs="?")
    parser.add_argument("--load-data", nargs="?", default=False, type=bool)
    parser.add_argument("--gpu-memory", nargs="?", default=128, type=int)
    parser.add_argument("--num-training-sets", nargs="?", default=2, type=int)
    parser.add_argument("--save-prefix", nargs="?")
    parser.add_argument("--nonlinearity", nargs="?")
    parser.add_argument("--swish-beta", nargs='?', default=1.0, type=float)
    parser.add_argument("--num-samples", nargs="?", default=100000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=30000, type=int)
    parser.add_argument("--num-data", nargs='?', default=1000, type=int)
    parser.add_argument("--num-hidden", nargs='?', default=4, type=int)

    parser.add_argument("--num-hidden-true", nargs='?', default=0, type=int)
    parser.add_argument("--prior-sd", nargs='?', default=1.0, type=float)
    parser.add_argument("--x-max", nargs='?', default=1, type=int)
    parser.add_argument("--target-accept-prob", nargs='?', default=0.8, type=float)
    parser.add_argument("--num-betas", default=8, type=int)

    # old mc_num_results = num_samples
    # old mc_burnin_steps = num_warmup
    # old training_sample_size = num_data
    # old global_prior_weight_std = prior_sd
    # old mc_target_accept = target_accept_prob
    # old num_betas = num_betas
    # old num_hidden_nodes = num_hidden
    # old symmetry_factor = num_hidden_true
    # old num_training_sets = num_training_sets
    # old nonlinearity = nonlinearity
    # old swish_beta = swish_beta
    
    args = parser.parse_args()
    
    args_dict = vars(args)
    print(args_dict)

    np.random.seed()
    
    # n = args.num_data
    betas = np.linspace(1 / np.log(args.num_data) * (1 - 1 / np.sqrt(2 * np.log(args.num_data))),
                                     1 / np.log(args.num_data) * (1 + 1 / np.sqrt(2 * np.log(args.num_data))),  args.num_betas)
                                     
    multiprocessing.set_start_method('spawn') # seems necessary to get multiprocessing to work well with TF (https://github.com/tensorflow/tensorflow/issues/5448)
    
    manager = Manager()
    samples = manager.dict()
    timers = manager.dict()
    datasets = manager.dict()
    jobs = []

    for i in range(args.num_betas):
        print("Beta [" + str(i+1) + "/" + str(args.num_betas) + "] " + str(betas[i]))

        for j in range(args.num_training_sets):
            print("       Dataset [" + str(j+1) + "/" + str(args.num_training_sets) + "]")

            # Attempt to load from disk
            filename = args.save_prefix + '/' + args.experiment_id + '-beta' + str(i) + '-dataset' + str(j) + '.pickle'

            if os.path.isfile(filename) and args.load_data:
                with open(filename, 'rb') as handle:
                    fine_chain, fine_trace = pickle.load(handle)
                samples[(i,j)] = [fine_chain, fine_trace]
                timers[(i,j)] = 0
                print("         Loaded chain from disk")
            else:
                print("         Spawning job...")
                p = Process(target=run_chains, args=(betas[i], i, j, samples, timers, datasets, args))
                jobs.append(p)
                p.start()
    
    for p in jobs:
        p.join()
        
    # Post processing  
    print("\n[ Post-processing ]\n")
    
    args_filename = args.save_prefix + '/' + args.experiment_id + '-args.pickle'
    with open(args_filename, 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    for i in range(args.num_betas):
      b = betas[i]
      print("Beta [" + str(i) + "/" + str(args.num_betas) + "] " + str(b))
      
      for j in range(args.num_training_sets):
        print("       Dataset [" + str(j) + "/" + str(args.num_training_sets) + "]")
        sample_filename = args.save_prefix + '/' + args.experiment_id + '-beta' + str(i) + '-dataset' + str(j) + '.pickle'
        dataset_filename = args.save_prefix + '/' + args.experiment_id + '-beta' + str(i) + '-dataset' + str(j) + '-train.pickle'
        
        # Save this chain to disk
        with open(sample_filename, 'wb') as handle:
          pickle.dump(samples[(i,j)], handle, protocol=pickle.HIGHEST_PROTOCOL)
          print("         Saved chain to disk")
          
        with open(dataset_filename, 'wb') as handle:
          pickle.dump(datasets[(i,j)], handle, protocol=pickle.HIGHEST_PROTOCOL)
          print("         Saved dataset to disk")
          
        print("         Time taken (s):", timers[(i,j)])

        fine_chain, fine_trace = samples[(i,j)]
        
        print("         Acceptance rate:", fine_trace.is_accepted[-args.num_samples:].numpy().mean())
        print("         Step size:", np.asarray(fine_trace.step_size[-args.num_samples:]).mean())
        num_divergences = fine_trace.has_divergence[-args.num_samples:].numpy().sum()
        print("         Divergences: {0}/{1} = {2}".format(num_divergences,args.num_samples,num_divergences/args.num_samples))  
# https://stackoverflow.com/questions/50937362/multiprocessing-on-python-3-jupyter
