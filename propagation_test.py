import scipy.special as sp
import numpy as np


def mass(alpha, beta, n):
    k = n // 2
    r = n % 2
    s = 0
    for i in range(k+1):
        s += sp.binom(9, k-i) * sp.binom(9-(k-i), r+2*i) * (alpha*beta)**(k-i) * (alpha+beta-2*alpha*beta)**(r+2*i) *\
             ((1-alpha)*(1-beta)) ** (9-k-r-i)
    return s


if __name__ == "__main__":
    for alpha in np.linspace(0, 1, 10):
        for beta in np.linspace(0, 1, 10):
            m = [mass(0.5, 0.6, n) for n in range(19)]
            if abs(np.sum(m) - 1.0) > 0.001:
                print(m)
                print(alpha, beta)
                print(np.sum(m))