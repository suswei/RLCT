import wandb
import RLCT

sweep_config = {
  "name": "My Sweep",
  "method": "grid",
  "parameters": {
        "betalogscale": {
            "values": [True,False]
        },
        "dataset": {
            "values": ['breastcancer-binary','iris-binary']
        },
        "batchsize": {
            "values": [10]
        },
        "network": {
            "values": ['logistic']
        },
        "epochs": {
            "values": [100]
        }
    }
}

# initialize the sweep
sweep_id = wandb.sweep(sweep_config)

# run the sweep agent
# TODO: encountering module object is not callabale error
wandb.agent(sweep_id,function=RLCT)
