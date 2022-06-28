# KPV

This depository contains implementation of Kernel Proxy Variable (KPV) approach, the first method suggested in 'Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction' (https://arxiv.org/pdf/2105.04544.pdf). 
More details and further simulations can be found in https://github.com/Afsaneh-Mastouri/KPV

# How to run the code?

The main accepts the observerd sample in form of dictionaries with seperate label for training (for calculation of stage 1 and stage 2) and test (calculationof causal effect based on causal function estimated using training data). To run the code you need to:

copy/download main + utils + cal_alpha
Add path/address of True_Caual_Effect.npz to load #do_cal at line #66 of main.py
Add path/address of Data_Sample.npz to load samples at line #71 of main.py Results are compressed and saved at the same directory/path as the main.py.

# Acknowledgments

The initial implementation of the code based on step by step calculation of causal effect (according to Proposition 2 of paper) was slow in estimating causal effect from large samples n>1000. 
Thanks to Liyuan Xu, we have improved implementation by replacing the final step of calculating alpha by the function utils.stage2_weights (nice trick! thank you Liyuan @liyuan9988).
You can see Liyuan full code at https://github.com/liyuan9988/DeepFeatureProxyVariable
