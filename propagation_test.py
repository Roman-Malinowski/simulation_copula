import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import norm, multivariate_normal

plt.rcParams['text.usetex'] = True


def c_min(l: np.array):
    return min(l)


def c_prod(l: np.array):
    return np.product(l)


def c_gaussian(l_: np.array, r=None):
    if 0.0 in l_:
        return 0.0
    if (l_ == 1.0).all():
        return 1.0

    l = l_[l_ != 1.]
    if len(l) == 1:
        return l[0]
    if not r:
        r = np.eye(len(l_))
    r = r[l_ != 1.][:, l_ != 1.]
    r[r == 0] = 0.5

    dist = multivariate_normal(cov=r)
    c = dist.cdf(np.array([norm.ppf(i) for i in l]))

    return c


def delta_c(l: list, d: dict, c=c_prod):
    s = 0
    d_cumul = {i: np.sum([d[str(k)] for k in range(int(i)+1)]) for i in d.keys()}
    l_key = [[k for k in d.keys() if d[k] == i][0] for i in l]

    for i in range(10):
        sub_volumes = [np.array(k) for k in combinations(range(9), i)]
        for x in sub_volumes:
            sub = np.array([d_cumul[k] for k in l_key])
            if len(x) > 0:
                sub[x] = sub[x] - np.array(l)[x]
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
        # s += sp.binom(9, k-i) * sp.binom(9-(k-i), r+2*i) * (alpha*beta)**(k-i) * (alpha+beta-2*alpha*beta)**(r+2*i) * ((1-alpha)*(1-beta)) ** (9-k-r-i)
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

    m_m = [mass(a, b, -n, c=c_min) for n in range(-18, 1)]
    cs_m = np.cumsum(m_m)
    pi_m = np.hstack([cs_m, np.flip(cs_m[:-1])])
    plt.plot(range(459 - 18, 459 + 19), pi_m, 'r+')

    m_g = [mass(a, b, -n, c=c_gaussian) for n in range(-18, 1)]
    cs_g = np.cumsum(m_g)
    pi_g = np.hstack([cs_g, np.flip(cs_g[:-1])])
    plt.plot(range(459 - 18, 459 + 19), pi_g, 'g+')
    print(m_g)
    print(pi_g)

    plt.title("$\pi$")
    plt.xlim(459-19, 459+19)
    plt.ylim(-0.01, 1.01)
    plt.legend(['$\Pi$', '$\min$'])
    plt.xticks([459+5*i for i in range(-3, 4)], ["$"+str(459+5*i)+"$" for i in range(-3, 4)])

    plt.show()

