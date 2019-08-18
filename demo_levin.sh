#! bash

# Due to random perturbations to input of G_x, the results may be slightly different from those in the paper. 
# 5000 iterations
python selfdeblur_levin_iter5k.py 

# 20000 iterations usually lead to better results than those in the paper. 
python selfdeblur_levin_iter20k.py 

# Given our learned deep models, we can directly reproduce the results reported in the paper. 
# python selfdeblur_levin_nonblind.py  
