# Bayesian-Optimization

In mr_histo.py I want to find the maximum of hidden_funct by creating a gaussian process with a RBF kernel. The acquisition
function is 'upper confidence bonds' with 1 as hyperparamter (completely arbitrary and not optimized). At every cycle of the loop
the gp is fitted with new explored data at the previous cycle, while the acquition function predicts whereis the expected next
maximum.

To run this script need sklearn and the bayesian optimization package found at this link
https://github.com/fmfn/BayesianOptimization
