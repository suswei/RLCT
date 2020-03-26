# Notes on TensorFlow probability

Assume `import tensorflow_probability as tfp` and `tfd = tfp.distributions` in what follows. You create a distribution with e.g. `p = tfd.Normal(0.0,1.0)`. Some of the standard methods

* `p.log_prob(x)` ([source](https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/distributions/distribution.py#L871-L883)) Log probability density/mass function, returns log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type `self.dtype`. For example, in the above example this would return `-1/2x^2 - const`.

* `p.sample(shape)` where `shape` is `0D or 1D int32 Tensor.` Shape of the generated samples. For example `weight_prior.sample((a,b))` will generate a tensor of samples of shape [a,b].
