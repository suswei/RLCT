# Experiment log

Experiment notebooks are logged in `/experiments`.

## Running



## Completed

In reverse chronological order in terms of completion time

* **10-5-2020-C** on Grumble

```
global_prior_weight_std = 0.05
global_prior_bias_std = 0.05
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=5000
mc_adaptation_steps=4000
mc_num_results=10000
```
Same experiment as `10-5-2020-B`. Again **RLCT robust estimate was 0.0**.

* **10-5-2020-B** on Grumble

```
global_prior_weight_std = 0.05
global_prior_bias_std = 0.05
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=5000
mc_adaptation_steps=4000
mc_num_results=10000
```
Same experiment as `10-5-2020-A` except that the global prior is now more localised (`std dev = 0.05`). The Markov chains were less well mixed than the original prior, and **RLCT robust estimate was 0.0**.

* **10-5-2020-A** on Grumble

```
global_prior_weight_std = 0.1
global_prior_bias_std = 0.1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=5000
mc_adaptation_steps=4000
mc_num_results=10000
```

Same experiment as `9-5-2020-C`,`9-5-2020-B`,`9-5-2020-A`,`8-5-2020-B`. Robust estimate of **RLCT = 0.12157140397173563**.

* **9-5-2020-C** on Grumble

```
global_prior_weight_std = 0.1
global_prior_bias_std = 0.1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=5000
mc_adaptation_steps=4000
mc_num_results=10000
```

Same experiment as `9-5-2020-B`,`9-5-2020-A`,`8-5-2020-B`. Robust estimate of **RLCT = 0.6687167531424518**.

* **9-5-2020-B** on Grumble

```
global_prior_weight_std = 0.1
global_prior_bias_std = 0.1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=5000
mc_adaptation_steps=4000
mc_num_results=10000
```

Same experiment as `9-5-2020-A`,`8-5-2020-B`. Robust estimate of **RLCT = -1.208686642703921**.

* **9-5-2020-A** on Grumble

```
global_prior_weight_std = 0.1
global_prior_bias_std = 0.1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=5000
mc_adaptation_steps=4000
mc_num_results=10000
```

Same experiment as `8-5-2020-B`. Robust estimate of **RLCT = 0.46887382556375895**.

* **8-5-2020-B** on Grumble. 

```
global_prior_weight_std = 0.1
global_prior_bias_std = 0.1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=5000
mc_adaptation_steps=4000
mc_num_results=10000
```

MC chains seemed in good health. Robust estimate of **RLCT = 0.1903613511780412**.
