# Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction 
In this work, we propose two kernel methods for causal effect estimation with proxy variables. 
The repository contains implementation for the algorithms to reproduce results in the Section 4 Experiments of the [paper](https://arxiv.org/abs/2105.04544).

## Installation
To install the PMMR package, run:
```
python setup.py sdist bdist_wheel
pip install -e .
```

To install dependencies, run:
```
pip install requirements.txt
```

## Rerunning experiments
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

KPV experiments - TBC

## Summaries and plots
To reproduce the results summaries and plots, run:
```
python post_process_res_std.py 
python post_process_res_ab.py 
python post_process_res_edu.py --sem=edu_IM_80
python post_process_res_edu.py --sem=edu_IR_80
```
The latex code for the tables of results are found in `results/sim_1d_no_x`, and the plots are found in `results/sim_1d_no_x/plots/`.
