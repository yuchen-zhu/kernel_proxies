# Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction
In this work, we propose two kernel methods for causal effect estimation with proxy variables. 
The repository contains implementation for the algorithms to reproduce results in the Section 4 Experiments of the [paper](https://arxiv.org/abs/2105.04544).

To install the PMMR package, run:
```
python setup.py sdist bdist_wheel
pip install -e .
```
 
To run experiments for PMMR, run:
```
python PMMR/pmmr_rkhs_nystr.py --sem=std --hparam=lmo --lmo='albl'
python PMMR/pmmr_rkhs_nystr_ab.py --sem=ab --hparam=cube
python PMMR/pmmr_rkhs_nystr_edu.py --sem=edu_IM_80 --hparam=cube
```
for synthetic, abortion and criminality, and education-grade-retention datasets respectively.

To reproduce the results plots, run:
(tbc)
