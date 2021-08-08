# Arthur simulation
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# parser = argparse.ArgumentParser(description='settings for simulation')
# parser.add_argument('--sem', type=str, help='sets SEM')
#
# args = parser.parse_args()

# number of samples
N = 5000

PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
sem = 'arthur'
seed = 2

np.random.seed(seed)

u2 = np.random.rand(N, 1) * 3 - 1
u1 = np.random.rand(N, 1) - ((u2 > 0) & (u2 < 1)).astype(int)

plt.plot(u1, u2, 'b.')
plt.xlabel('u1')
plt.ylabel('u2')

os.makedirs(PATH + sem + '_seed' + str(seed) + '/', exist_ok=True)
plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'u1u2.png'), plt.close()

aStd = 0.05
a = u2 + np.random.randn(*u2.shape) * aStd

y = u2 * np.cos(2 * (a + 0.3 * u1 + 0.2))

##########################
# backdoor: ground truth #
##########################

# for each a, generate lots of u, marginalise
naxisA = 100
aAx = np.linspace(np.round(np.min(a)), np.round(np.max(a)), naxisA).T

nVal = 10000  # number of validation samples
yBkD = np.zeros((naxisA, 1))

for indA in range(naxisA):
    u2_BkD = np.random.rand(nVal, 1) * 3 - 1
    u1_BkD = np.random.rand(nVal, 1) - ((u2_BkD > 0) & (u2_BkD < 1))
    yBkD[indA] = np.mean(u2_BkD * np.cos(2 * (aAx[indA]) + 0.3 * u1_BkD + 0.2))

#################################
# Marginalisation: ground truth #
#################################

yMar = np.zeros((naxisA, 1))

for indA in range(naxisA):
    u2_mar = aAx[indA] + np.random.randn(nVal, 1) * aStd
    u1_mar = np.random.rand(nVal, 1) - ((u2_mar > 0) & (u2_mar < 1))

    # debug: check that conditioning is correct
    if 0:
        aAx[indA]
        plt.clf()
        plt.plot(u1_mar, u2_mar, '.')
        plt.plot(u1, u2, 'r.')
        plt.show()

    yMar[indA] = np.mean(u2_mar * np.cos(2 * (aAx[indA])) + 0.3 * u1_mar + 0.2)

plt.plot(aAx, yBkD, label='backdoor')
plt.plot(aAx, yMar, label='marginalisation')
plt.legend()
plt.savefig(PATH + sem + '_seed' + str(seed) + '/' + 'bkd-marg.png'), plt.close()

