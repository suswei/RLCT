import wandb
import RLCT

# initialize the sweep
sweep_id = wandb.sweep('sweep.yaml')

# run the sweep agent
wandb.agent(sweep_id,function=RLCT)
