import numpy as np


#######################
## Global parameters ##
#######################
# constants
slip = 0.1
guess = 0.1

last_n = 2
n_skills = 2
n_units = 2
c = np.random.uniform(size=(n_units, n_skills))
c = c/c.sum(axis=-1).reshape(-1, 1)
c_exam = np.random.uniform(size=(n_skills, ))
c_exam = c_exam / c_exam.sum(axis=-1)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# Prior on U1
def prior(n_skills):
    np.random.seed(2)
    'Prior on u1 - the vector of skills. Try multivariate normal.'
    return np.random.multivariate_normal(np.zeros((n_skills)), cov=np.eye(n_skills))

# Observation / Proxy
def leftprox(u1):
    'Left proxy is the last n scores.'
    last = []
    for i in range(last_n):
        unit = np.random.choice(np.arange(n_units))
        weights = c[unit,:]
        learnt = weights.dot(sigmoid(u1))
        corr = learnt*(1-slip) + (1-learnt)*guess
        last.append(min(max(corr + np.random.normal(0, scale=0.1), 0), 1))
    z = last
    return z

def rightprox(u):
    'Right proxy is the average score in each unit.'
    av_scores_raw = c.dot(sigmoid(u)) + np.random.normal(0, scale=0.01, size=(n_units, )) # just use a smaller variance for now
    av_scores = []
    for sc in av_scores_raw:
        av_scores.append(min(max(sc, 0), 1))
    av_scores = np.array(av_scores)
    return av_scores

# Behaviour Policy
def pi_b(u1, z, gamma1=0.1):
    'Behaviour policy as a function of u1 and z. gamma is a hyperparameter controlling the effect of confidence.'
    row_max_mask = np.tile(c.max(axis=1).reshape(-1,1), [1, c.shape[1]]) == c
    c_noise = c * row_max_mask + np.random.normal(0, scale=1, size=(n_units, ))
#     print(c_noise)

    return np.tanh(np.exp(-2 * c_noise.dot(u1))) + gamma1 * np.exp(-np.mean(z))

# Target Policy
def pi_e(u1, z):
    pass

# Transition
def transition(u1, a, gamma2=0):
    'Transition function from u1 and a to u2.'
    u2 = np.log(np.exp(u1) + a) + gamma2 * np.random.multivariate_normal(np.zeros((u1.shape[0],)), cov=np.eye(u1.shape[0]))
    return u2

# Outcome / Reward
def reward(u2, a):
    'Reward as a function of u2.'
    return min(max(np.tanh(np.mean(u2)) + 0.01*np.random.normal(), 0), 1)
