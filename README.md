## Reproducing generalization experiments

Run lastlayer_sweep.py for all 32 taskids. Results are dumped to lastlayersims folder. 
To each taskid there are three files associated
+ taskid*_args.pt: contains all the arguments used in lastlayerbayesian.py
+ taskid*_results.pt: contains generalization error for each method and each sample size
+ taskid*.png: preliminary graphic, not publication pretty
Run lastlayer_visualize.py to produce pretty learning curves and latex tables summarizing learning coefficient and R2 fit.

The random seed was set to 43 in lastlayerbayesian.py.

## Reproducing symmetry experiments