# Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction 
In this work, we propose two kernel methods for causal effect estimation with proxy variables. 
The repository contains implementation for the algorithms to reproduce results in the Section 4 Experiments of the [paper](https://arxiv.org/abs/2105.04544).

## KPV

This depository contains implementation of Kernel Proxy Variable (KPV) approach, the first method suggested in 'Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction' (https://arxiv.org/pdf/2105.04544.pdf). 
More details and further simulations can be found in https://github.com/Afsaneh-Mastouri/KPV

### How to run the code?

The main accepts the observerd sample in form of dictionaries with seperate label for training (for calculation of stage 1 and stage 2) and test (calculationof causal effect based on causal function estimated using training data). To run the code you need to:

1. copy/download main + utils + cal_alpha
2. Add path/address of True_Caual_Effect.npz to load #do_cal at line #66 of main.py. 
3. Add path/address of Data_Sample.npz to load samples at line #71 of main.py Results are compressed and saved at the same directory/path as the main.py.


## PMMR

### Installation
To install the PMMR package, run:
```
python setup.py sdist bdist_wheel
pip install -e .
```

To install dependencies, run:
```
pip install requirements.txt
```

### Rerunning experiments
For baseline experiments, run:
```
python baselines_scripts/run_zoo_experiments_more_baselines.py --sem=std
python baselines_scripts/run_ab_experiments_more_baselines.py --sem=ab
python baselines_scripts/run_edu_experiments_more_baselines.py --sem=edu_IM_80
python baselines_scripts/run_edu_experiments_more_baselines.py --sem=edu_IR_80
```
 
For PMMR experiments, run:
```
python PMMR/pmmr_rkhs_nystr.py --sem=std --hparam=lmo --lmo=albl
python PMMR/pmmr_rkhs_nystr_ab.py --sem=ab --hparam=cube
python PMMR/pmmr_rkhs_nystr_edu.py --sem=edu_IM_80 --hparam=cube
python PMMR/pmmr_rkhs_nystr_edu.py --sem=edu_IR_80 --hparam=cube
```
for synthetic, abortion and criminality, and education-grade-retention datasets respectively.

KPV experiments - See KPV directory or https://github.com/Afsaneh-Mastouri/KPV

### Summaries and plots
To reproduce the results summaries and plots, run:
```
python post_process_res_std.py 
python post_process_res_ab.py 
python post_process_res_edu.py --sem=edu_IM_80
python post_process_res_edu.py --sem=edu_IR_80
```
The latex code for the tables of results are found in `results/sim_1d_no_x`, and the plots are found in `results/sim_1d_no_x/plots/`.

## Acknowledgments

- The initial implementation of KPV based on step by step calculation of causal effect (according to Proposition 2 of paper) was slow in estimating causal effect from large samples n>1000. 
Thanks to Liyuan Xu, we have improved implementation by replacing the final step of calculating alpha by the function utils.stage2_weights (nice trick! thank you Liyuan @liyuan9988).
You can see Liyuan full code at https://github.com/liyuan9988/DeepFeatureProxyVariable

- We gratefully acknowledge [Rui Zhang](https://github.com/RuiZhang2016/MMRIV) for his repo on MMRIV, which we built our code base on.

