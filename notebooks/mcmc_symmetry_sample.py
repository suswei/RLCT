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
import ray

#TraceReturns = collections.namedtuple('TraceReturns', ['target_log_prob',
#                                                   'leapfrogs_taken',
#                                                   'has_divergence',
#                                                   'energy',
#                                                   'log_accept_ratio',
#                                                    'is_accepted',
#                                                      'step_size'])
                                                          
# For using Ray with TensorFlow see https://docs.ray.io/en/latest/using-ray-with-tensorflow.html
@ray.remote(num_cpus=1,num_gpus=0.05)
def run_chains(beta_index, dataset_index, args):   
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

    betas = np.linspace(1 / np.log(args.num_data) * (1 - 1 / np.sqrt(2 * np.log(args.num_data))),
                                     1 / np.log(args.num_data) * (1 + 1 / np.sqrt(2 * np.log(args.num_data))),  args.num_betas)
    beta = betas[beta_index]
                                     
    # https://github.com/tensorflow/probability/blob/7cf006d6390d1d0a6fe541e5c49196a58a22b875/tensorflow_probability/python/mcmc/nuts.py#L86

    def trace_fn_nuts(_, pkr):
        return {
            "target_log_prob": pkr.inner_results.inner_results.target_log_prob,
            "leapfrogs_taken": pkr.inner_results.inner_results.leapfrogs_taken,
            "has_divergence": pkr.inner_results.inner_results.has_divergence,
            "energy": pkr.inner_results.inner_results.energy,
            "log_accept_ratio": pkr.inner_results.inner_results.log_accept_ratio,
            "is_accepted": pkr.inner_results.inner_results.is_accepted,
            "step_size": pkr.inner_results.inner_results.step_size,
        }
    
    def nest_concat(*args):
        return tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=0), *args)
      
    @tf.function(autograph=False,experimental_compile=True)
    def graph_hmc(*args, **kwargs):
        return tfp.mcmc.sample_chain(*args, **kwargs)
        
    def run_nuts(
        target_log_prob_fn,
        inits,
        bijectors_list=None,
        num_burnin_steps=1000,
        num_results=1000,
        num_chains=1,
        checkpoint=1):

        assert num_results % checkpoint == 0
        
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
        
        kernel_results = kernel.bootstrap_results(inits)
        current_state = inits
        num_blocks = int(num_results/checkpoint)
        
        # we divide num_results into blocks, each of size checkpoint
        # only use burn-in for the first block
        sample_rates = []
        
        for i in range(num_blocks):
            c = 0 if i > 0 else 1
            
            block_filename = args.save_prefix + '/' + args.experiment_id + '-beta' + str(beta_index) + '-dataset' + str(dataset_index) + '-block' + str(i) + '.pickle'
            
            # If the file for this block exists, load the current_state and kernel_results
            if os.path.isfile(block_filename):
                with open(block_filename, 'rb') as handle:
                    chain, trace, kernel_results = pickle.load(handle)
            
                block_str = "Beta [{0}] Dataset [{1}] block [{2}/{3}] save data found, SKIPPING".format(beta_index+1,dataset_index+1,i+1,num_blocks)
                print(block_str)
            else:
                block_pre_time = time.time()
                
                chain, trace, kernel_results = graph_hmc(
                    num_results=checkpoint,
                    num_burnin_steps=num_burnin_steps*c,
                    current_state=current_state,
                    kernel=kernel,
                    trace_fn=trace_fn_nuts,
                    num_steps_between_results=args.thin_factor,
                    previous_kernel_results=kernel_results,
                    return_final_kernel_results=True
                )
            
                block_post_time = time.time()
                block_delta_str = '{0:.2f}'.format(block_post_time - block_pre_time)
                sample_rate = (checkpoint + num_burnin_steps*c)/(block_post_time - block_pre_time)
                block_samplerate_str = '{0:.2f}'.format(sample_rate)
                if( num_burnin_steps*c == 0 ): sample_rates.append(sample_rate)
                                            
                # Save this block to disk
                with open(block_filename, 'wb') as handle:
                    pickle.dump([chain,trace,kernel_results], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    block_str = "Beta [{0}] Dataset [{1}] block [{2}/{3}] saved to disk ".format(beta_index+1,dataset_index+1,i+1,num_blocks)
                    block_str = block_str + "[" + block_delta_str + "s " + block_samplerate_str + " samples/s]"
                    print(block_str)
            
            current_state = tf.nest.map_structure(lambda x: x[-1], chain)
            del chain
            del trace
            
        return sample_rates

    true_dist_state = mcmc_symmetry.true_distribution(args.num_hidden_true,args.num_hidden)

    def swish(X):
        return 1/args.swish_beta * tf.nn.swish( args.swish_beta * X )
        
    if( args.nonlinearity == "relu" ):
        nl = tf.nn.relu
    else:
        nl = swish
            
    true_network = mcmc_symmetry.build_network(true_dist_state[::2],
                                 true_dist_state[1::2], nl)

    # Generating training data
    X_train, y_train = mcmc_symmetry.generate_data(args.num_data, 
                                                    true_network, 
                                                    args.x_max)

    center = true_dist_state

    #NUM_PARAMS = 0
    #for s in true_dist_state:
      #print("State shape", s.shape)
    #  NUM_PARAMS += s.shape.num_elements()
    #print("Total params", NUM_PARAMS)

    weight_prior = tfd.Normal(0.0, args.prior_sd)

    # NOTE: to get XLA to work I needed to make a symlink
    # https://github.com/google/jax/issues/989

    #if( args.use_exchange ):
    #    run_nuts_opt = tf.function(run_nuts_emc,autograph=False,experimental_compile=True)
    #    joint_log_prob_fn = mcmc_symmetry.joint_log_prob_fn_emc
    #    center_mc = center
    #else:
    #run_nuts_opt = run_nuts #tf.function(run_nuts,autograph=False,experimental_compile=True)
    joint_log_prob_fn = mcmc_symmetry.joint_log_prob_fn
    center_mc = [tf.expand_dims(x,axis=0) for x in center]

    logp = partial(joint_log_prob_fn, center, weight_prior,
                        X_train, y_train, beta, nl)

    pre_time = time.time()
  
    sample_rates = run_nuts(target_log_prob_fn=logp,
                inits=center_mc,
                num_burnin_steps=args.num_warmup,
                num_results=args.num_samples,
                checkpoint=args.checkpoint)

    post_time = time.time()
    delta_time = post_time - pre_time

    # For processing of EnLn estimates
    log_prob_noprior = partial(
        mcmc_symmetry.joint_log_prob_fn, center, None, X_train, y_train, 1.0, nl
    )
    
    # Do post processing
    num_blocks = int(args.num_samples/args.checkpoint)
    
    chain_blocks = []
    trace_blocks = []
    
    for i in range(num_blocks):
        block_filename = args.save_prefix + '/' + args.experiment_id + '-beta' + str(beta_index) + '-dataset' + str(dataset_index) + '-block' + str(i) + '.pickle'
                                        
        # Save this block to disk
        with open(block_filename, 'rb') as handle:
            chain, trace, _ = pickle.load(handle)
            chain_blocks.append(chain)
            trace_blocks.append(trace)
    
    full_chain = tf.nest.map_structure(lambda *parts: tf.concat(parts,axis=0), *chain_blocks)
    full_trace = tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=0), *trace_blocks)
            
    # nL_n(w) = - log_prob_noprior(*w)
    n_Ln_samples = []

    # compute E_w[nL_n(w)] by averaging over the chain
    for y in range(len(full_chain[0])):
      w = [q[y] for q in full_chain]
      n_Ln_samples.append(-log_prob_noprior(*w))
  
    n_Ln_mean = np.mean(n_Ln_samples)
    
    return full_chain, full_trace, n_Ln_mean, delta_time, sample_rates

if __name__ == '__main__':                                
    parser = argparse.ArgumentParser(description="RLCT_HMC_symmetric")
    parser.add_argument("--experiment-id", nargs="?")
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
    parser.add_argument("--thin-factor", default=1, type=int)
    parser.add_argument("--checkpoint",default=0, type=int)

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
    
    if( args.checkpoint == 0 ): args.checkpoint = args.num_samples
    
    args_dict = vars(args)
    print(args_dict)

    args_filename = args.save_prefix + '/' + args.experiment_id + '-args.pickle'
    with open(args_filename, 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    np.random.seed()
    
    # n = args.num_data
    betas = np.linspace(1 / np.log(args.num_data) * (1 - 1 / np.sqrt(2 * np.log(args.num_data))),
                                     1 / np.log(args.num_data) * (1 + 1 / np.sqrt(2 * np.log(args.num_data))),  args.num_betas)
                       
    # For Ray basics: https://docs.ray.io/en/latest/walkthrough.html
    ray.init()
    
    jobs = [(run_chains.remote(i,j,args),i,j) for i in range(args.num_betas) 
                                            for j in range(args.num_training_sets)]

    overall_sample_rates = []
    
    for (obj_ref, i, j) in jobs:
        full_chain, full_trace, n_Ln_mean, delta_time, sample_rates = ray.get(obj_ref)    
        overall_sample_rates += sample_rates
        
        print("Beta [" + str(i+1) + "/" + str(args.num_betas) + "] Dataset [" + str(j+1) + "/" + str(args.num_training_sets) + "]")
        sample_filename = args.save_prefix + '/' + args.experiment_id + '-beta' + str(i) + '-dataset' + str(j) + '.pickle'
        EnLn_filename = args.save_prefix + '/' + args.experiment_id + '-beta' + str(i) + '-dataset' + str(j) + '-estimate.pickle'
        time_filename = args.save_prefix + '/' + args.experiment_id + '-beta' + str(i) + '-dataset' + str(j) + '-time.pickle'
        
        # Save this chain to disk
        with open(sample_filename, 'wb') as handle:
            pickle.dump([full_chain,full_trace], handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("         Saved chain to disk")

        with open(EnLn_filename, 'wb') as handle:
            pickle.dump(n_Ln_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("         Saved EnLn estimate to disk")
            
        with open(time_filename, 'wb') as handle:
            pickle.dump(delta_time, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("         Time taken: {0:.2f}s".format(delta_time))
        print("         Acceptance rate:", full_trace["is_accepted"][-args.num_samples:].numpy().mean())
        print("         Step size:", np.asarray(full_trace["step_size"][-args.num_samples:]).mean())
        num_divergences = full_trace["has_divergence"][-args.num_samples:].numpy().sum()
        print("         Divergences: {0}/{1} = {2}".format(num_divergences,args.num_samples,num_divergences/args.num_samples))  
    
    if( len(overall_sample_rates) > 0 ):
        print("Mean sample rate: {0:.2f} samples/s".format(np.mean(overall_sample_rates)))
    ray.shutdown()