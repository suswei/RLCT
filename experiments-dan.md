# Experiment log

Experiment notebooks are logged in `/experiments` on the Dropbox. 

```
Mean distances
==============

ED = expected distance of prior

Experiment       Prior     ED        Actual
-------------------------------------------
11-5-2020-A      0.1       0.361     0.376
10-5-2020-J      0.1       0.361     0.347
10-5-2020-I      1.0       3.606     3.428
10-5-2020-H      1.0       3.606     3.363
10-5-2020-G      0.05      0.180     0.175
10-5-2020-F      0.05      0.180     0.173
10-5-2020-E      0.05      0.180     0.167
10-5-2020-D      0.05      0.180     0.173
10-5-2020-C      0.05      0.180     0.178
10-5-2020-B      0.05      0.180     0.183
10-5-2020-A      0.1       0.361     0.374
```

## Running

* **11-5-2020-C** on Grumble

```
num_betas = 20
global_prior_weight_std = 0.1
global_prior_bias_std = 0.1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=30000
mc_adaptation_steps=20000
mc_num_results=10000
```
Same experiment as `11-5-2020-A,10-5-2020-J` except with twice as many beta values.

## Completed

In reverse chronological order in terms of completion time

* **11-5-2020-B** on Grumble

```
global_prior_weight_std = 0.1
global_prior_bias_std = 0.1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=30000
mc_adaptation_steps=20000
mc_num_results=10000
```
Same experiment as `11-5-2020-A,10-5-2020-J`. **Robust estimate of RLCT = 0.8644861041992501**.

* **11-5-2020-A** on Grumble

```
global_prior_weight_std = 0.1
global_prior_bias_std = 0.1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=30000
mc_adaptation_steps=20000
mc_num_results=10000
```
Same experiment as `10-5-2020-J`. Robust estimate of **RLCT = 0.3969993277416406**.

* **10-5-2020-J** on Grumble

```
global_prior_weight_std = 0.1
global_prior_bias_std = 0.1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=30000
mc_adaptation_steps=20000
mc_num_results=10000
```
Back to `std=0.1` but now with more burnin and adaptation. ESS and other chain metrics look very healthy. **Robust estimate of RLCT = 0.2295389323186636**.

* **10-5-2020-I** on Grumble

```
global_prior_weight_std = 1
global_prior_bias_std = 1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=15000
mc_adaptation_steps=14000
mc_num_results=10000
```
Same experiment as `10-5-2020-H`. Robust estimate of **RLCT = 1.1879232373969846**.

* **10-5-2020-H** on Grumble

```
global_prior_weight_std = 1
global_prior_bias_std = 1
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=15000
mc_adaptation_steps=14000
mc_num_results=10000
```
Changing to `std=1`. Robust estimate of **RLCT = 0.352919132045753**.

* **10-5-2020-G** on Grumble

```
global_prior_weight_std = 0.05
global_prior_bias_std = 0.05
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=15000
mc_adaptation_steps=14000
mc_num_results=10000
```
Same experiment as `10-5-2020-F, 10-5-2020-E`. `min_ess = 0.00046881728`. **Robust estimate of RLCT = -0.14732375833049208**. It seems that we cannot fit to the posterior with this prior!

* **10-5-2020-F** on Grumble

```
global_prior_weight_std = 0.05
global_prior_bias_std = 0.05
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=15000
mc_adaptation_steps=14000
mc_num_results=10000
```
Same experiment as `10-5-2020-E`. `min_ess = 0.0005416768`. **Robust estimate of RLCT = 0.5991095777790726**.

* **10-5-2020-E** on Grumble

```
global_prior_weight_std = 0.05
global_prior_bias_std = 0.05
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=15000
mc_adaptation_steps=14000
mc_num_results=10000
```
Same experiment as `10-5-2020-D` but with even more burnin and adaptation steps. **Robust estimate of RLCT = 0.6344730623649024**.

* **10-5-2020-D** on Grumble

```
global_prior_weight_std = 0.05
global_prior_bias_std = 0.05
symmetry_factor = 3
training_sample_size = 20
num_training_sets = 3
mc_burnin_steps=10000
mc_adaptation_steps=8000
mc_num_results=10000
```
Same experiment as `10-5-2020-C` except that the number of burnin and adaptation steps was doubled. Now the ESS/step looks better and **RLCT robust estimate was 0.052440834179538466**.

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
