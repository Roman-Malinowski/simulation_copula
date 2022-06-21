import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import norm, multivariate_normal, random_correlation

plt.rcParams['text.usetex'] = True


def c_min(u: np.array):
    return min(u)


def c_prod(u: np.array):
    return np.product(u)


def c_gaussian(u: np.array, r=None):
    if r is None:
        r = np.eye(len(u))
    if 0.0 in u:
        # Sometimes having a 0 in l gives nan instead of 0...
        return 0
    dist = multivariate_normal(cov=r)
    c = dist.cdf(np.array([norm.ppf(i) for i in u]))
    return c


def random_correlation_matrix(size=9):
    # Create a random correlation matrix
    rng = np.random.default_rng(10)
    eigen_val = np.random.rand(size)
    eigen_val = eigen_val * size / sum(eigen_val)
    r = random_correlation.rvs(eigen_val)
    return r


def delta_c(l_mass: list, d: dict, c=c_prod):
    s = 0
    d_cumul = {i: np.sum([d[str(k)] for k in range(int(i)+1)]) for i in d.keys()}
    l_key = [[k for k in d.keys() if d[k] == i][0] for i in l_mass]

    for i in range(10):
        sub_volumes = [np.array(k) for k in combinations(range(9), i)]
        for x in sub_volumes:
            sub = np.array([d_cumul[k] for k in l_key])
            if len(x) > 0:
                sub[x] = sub[x] - np.array(l_mass)[x]
            s += (-1)**i*c(sub)
    return s


def mass(alpha, beta, n, c=c_prod, round_digit=10):
    d = {"0": round((1-alpha)*(1-beta), round_digit), "1": round(alpha+beta-2*alpha*beta, round_digit), "2": round(alpha*beta, round_digit)}

    k = n // 2
    r = n % 2

    s = 0
    for i in range(k+1):
        if 9-k-r-i < 0:
            continue

        h_vol = delta_c((k-i)*[round(alpha*beta, round_digit)] + (r+2*i)*[round(alpha+beta-2*alpha*beta, round_digit)] + (9-k-r-i)*[round((1-alpha)*(1-beta), round_digit)], d, c=c)
        s += sp.binom(9, k - i) * sp.binom(9 - (k - i), r + 2 * i) * h_vol
    return s


if __name__ == "__main__":
    a = 0.6
    b = 0.5

    plt.figure()
    m = [mass(a, b, -n) for n in range(-18, 1)]
    cs = np.cumsum(m)
    pi = np.hstack([cs, np.flip(cs[:-1])])
    plt.plot(range(459 - 18, 459 + 19), pi, 'b+')

    """m_m = [mass(a, b, -n, c=c_min) for n in range(-18, 1)]
    cs_m = np.cumsum(m_m)
    pi_m = np.hstack([cs_m, np.flip(cs_m[:-1])])
    plt.plot(range(459 - 18, 459 + 19), pi_m, 'r+')"""

    r = random_correlation_matrix()

    m_g = [mass(a, b, -n, c=lambda u: c_gaussian(u, r)) for n in range(-18, 1)]
    cs_g = np.cumsum(m_g)
    cs_g /= cs_g[-1]
    pi_g = np.hstack([cs_g, np.flip(cs_g[:-1])])
    plt.plot(range(459 - 18, 459 + 19), pi_g, 'gx')
    print(m_g)
    print(pi_g)

    plt.title(r"$\pi$")
    plt.xlim(459-19, 459+19)
    plt.ylim(-0.01, 1.01)
    plt.legend([r'$\Pi$', r'$Gaussian$'])
    plt.xticks([459+5*i for i in range(-3, 4)], ["$"+str(459+5*i)+"$" for i in range(-3, 4)])

    plt.show()

