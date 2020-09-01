# Library for MCMC-Symmetry
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
    
import multiprocessing
from multiprocessing import Process, Manager

def build_network_emc(weights_list, biases_list, nonlin):
    def dense(X, W, b):
        # temperature index t
        # batch index i
        # vector index x
        a = tf.einsum('itx,txy->ity',X,W)
        return nonlin(a + b)

    def model(X):
        # In the EMC case this is a bit more complicated, because the
        # incoming data has only a batch and vector index so we do the
        # first layer by hand
        a = tf.einsum('ix,txy->ity',X,weights_list[0])
        net = nonlin(a+biases_list[0])
        
        for (weights, biases) in zip(weights_list[1:-1], biases_list[1:-1]):
            net = dense(net, weights, biases)

        final_w = weights_list[-1]
        final_b = biases_list[-1]
        net = tf.einsum('itx,txy->ity',net,final_w) + final_b
        preds = net[:,:,0]
        
        return tfd.Normal(loc=preds, scale=1.0)

    return model
    
def build_network(weights_list, biases_list, nonlin):
    def dense(X, W, b):
        return nonlin(tf.matmul(X, W) + b)

    def model(X):
        net = X
        for (weights, biases) in zip(weights_list[:-1], biases_list[:-1]):
            net = dense(net, weights, biases)

        final_w = weights_list[-1]
        final_b = biases_list[-1]
        net = tf.matmul(net, final_w) + final_b
        preds = net[:, 0]

        return tfd.Normal(loc=preds, scale=1.0)

    return model

def true_distribution(symmetry_factor,num_hidden):
    """generate a symmetric true distribution
    Keyword Arguments:
        symmetry_factor -- number of sides of the polygon
        num_hidden -- number of nodes in the single hidden layer
    Returns:
        list -- architecture of FCNN with weigths and bias tensors for each layer
    """

    assert num_hidden >= symmetry_factor, "Number of hidden nodes must exceed symmetry factor"

    layers = (
        2,
        num_hidden,
        1,
    )

    a = 2 * np.pi / symmetry_factor
    t1 = np.array([[np.cos(a/2), np.sin(a/2)]])

    # The true distribution uses the beginning segment of the hidden nodes to encode
    # the hyperplanes bounding the polygon, and puts zeros for all other weights
    w_list = [ np.matmul(t1, np.array([[np.cos(k*a), -np.sin(k*a)],
                                             [np.sin(k*a), np.cos(k*a)]])) for k in range(symmetry_factor)]
    w_list.extend([ np.zeros_like(w_list[0]) for k in range(num_hidden-symmetry_factor)])
    w = np.vstack(w_list)

    w = np.transpose(w)
    b = np.concatenate([-0.3 * np.ones((symmetry_factor)), np.zeros((num_hidden-symmetry_factor))],axis=0)

    #q = np.transpose(np.vstack([np.ones((num_hidden)), np.zeros((num_hidden))]))
    q = np.concatenate([np.ones((symmetry_factor,1)), np.zeros((num_hidden-symmetry_factor,1))],axis=0)
    c = np.array([0.0])

    w_t = tf.constant(w,dtype=tf.float32)
    b_t = tf.constant(b,dtype=tf.float32)
    q_t = tf.constant(q,dtype=tf.float32)
    c_t = tf.constant(c,dtype=tf.float32)

    architecture = []
    architecture.extend((w_t,b_t))
    architecture.extend((q_t,c_t))

    return architecture
    
def generate_data(num_samples, true_network, bound):
    # The distribution q(x) is uniform on the unit square [-bound,bound]^2
    x1_coords = 2 * bound * np.random.rand(num_samples) - bound
    x2_coords = 2 * bound * np.random.rand(num_samples) - bound

    X_data = []
    y_data = []

    for (x1,x2) in zip(x1_coords, x2_coords):
        X = np.array([[x1, x2]])
        true_labels_dist = true_network(X.astype("float32"))
        y = true_labels_dist.sample().numpy()

        X_data.append(np.array([x1,x2]))
        y_data.append(y)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return (X_data, y_data)
    
def joint_log_prob_fn(center, weight_prior, X, y, beta, nonlin, *args):
    weights_list = args[::2]
    biases_list = args[1::2]

    center_weights = center[::2]
    center_biases = center[1::2]

    lp = 0.0

    # prior log-prob
    if( weight_prior != None ):
        lp += sum(
            [tf.reduce_sum(weight_prior.log_prob(w-cw))
             for (w,cw) in zip(weights_list,center_weights)])

        lp += sum([tf.reduce_sum(weight_prior.log_prob(b-bw))
              for (b,bw) in zip(biases_list,center_biases)])

    # likelihood of predicted labels
    network = build_network(weights_list, biases_list, nonlin)
    labels_dist = network(X.astype("float32"))
    lp += beta * tf.reduce_sum(labels_dist.log_prob(y))
    return lp

def joint_log_prob_fn_emc(center, weight_prior, X, y, beta, nonlin, *args):
    weights_list = args[::2]
    biases_list = args[1::2]

    center_weights = center[::2]
    center_biases = center[1::2]

    lp = 0.0
    
    print("\n")
    for (b,bw) in zip(biases_list,center_biases):
        print(b)
        print(bw)
        print("---")
    print("\n")
    
    # prior log-prob
    # first index is temperature
    if( weight_prior != None ):
        lp += sum(
            [tf.reduce_sum(weight_prior.log_prob(w-cw),axis=[1,2])
             for (w,cw) in zip(weights_list,center_weights)])

        lp += sum([tf.reduce_sum(weight_prior.log_prob(b-bw),axis=[1])
              for (b,bw) in zip(biases_list,center_biases)])
    
    # likelihood of predicted labels
    network = build_network_emc(weights_list, biases_list, nonlin)
    labels_dist = network(X.astype("float32"))
    print(labels_dist.log_prob(y))
    lp += beta * tf.reduce_sum(labels_dist.log_prob(y),axis=0)
    print(lp)
    return lp