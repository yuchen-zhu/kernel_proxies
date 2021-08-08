import numpy as np

#######################
## Global parameters ##
#######################

# constants
sed = 2


# Prior on U
def prior(N_u):
    np.random.seed(sed)
    'Prior on U - the vector of skills. Try multivariate normal.'
    return np.random.multivariate_normal(np.zeros((N_u)), cov=np.eye(N_u))


def gen_x(u):
    pass


def leftprox(u, x):
    'Left proxy is '


def rightprox(u, x):
    pass


def action(z, u, x):
    pass


def reward(a, w, x, u):
    pass


