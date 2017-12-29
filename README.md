# This repo has been adapted to the purpose of SVGD for time series

A simple extension of SVGD is used where the posterior distribution at each step 

p(x_t | y_t) is approximated using a set of SVGD particles. 

These particles are then passed through the transition density (similar to regular particle filtering)

p(x_{t+1}|x_{t})

where the process is repeated.


# Stein Variational Gradient Descent (SVGD)
SVGD is a general purpose variational inference algorithm that forms a natural counterpart of gradient descent for optimization. SVGD iteratively transports a set of particles to match with the target distribution, by applying a form of functional gradient descent that minimizes the KL divergence.

The package contains implementation of SVGD in both Matlab and Python language. Demos are also provided to reproduce the results in our paper. The Bayesian neural network example is based on [Theano](http://deeplearning.net/software/theano/).

For more information, please visit our project website - [SVGD](http://www.cs.dartmouth.edu/~dartml/project.html?p=vgd).

## Toy example on 1D Gaussian Mixture
Toy example with 1D Gaussian mixture. The red dashed lines are the target density function and the solid green lines are the densities of the particles at different iterations of our algorithm (estimated using kernel density estimator).

<img src="data/1dgmm.gif" width="50%">

## Basic Usage
```python
'''
  x0: initial particles
  dlnprob: returns first order derivative of log probability
  n_iter: number of iterations
  stepsize: initial learning rate 
'''
theta = SVGD().update(x0, dlnprob, n_iter, stepsize)
```

## Citation
Qiang Liu and Dilin Wang. [Stein Variational Gradient Descent (SVGD): A General Purpose Bayesian Inference Algorithm](http://arxiv.org/abs/1608.04471). NIPS, 2016.

## Feedback
Feedback is greatly appreciated. If you have any questions, comments, issues or anything else really, [shoot me an email](mailto:dilin.wang.gr@dartmouth.edu).

All rights reserved.
# ssvgd
