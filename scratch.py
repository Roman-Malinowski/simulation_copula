import numpy
import numpy as np
import warnings


def min_copula(u: float, v: float) -> float:
    """
    The minimun copula
    :param u: a float between 0 and 1
    :param v: a float between 0 and 1
    :return: a float between 0 and 1 corresponding to C(u,v)
    """
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s ; b=%s\\"
                      "Cropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    return min(u, v)


def lukaciewicz_copula(u: float, v: float) -> float:
    """
    The lukaciewicz copula
    :param u: a float between 0 and 1
    :param v: a float between 0 and 1
    :return: a float between 0 and 1 corresponding to C(u,v)
    """
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s ; b=%s\\"
                      "Cropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    return max(0., u + v - 1)


def ali_mikhail_haq_copula_pinf(a_j, a_i, b_j, b_i, theta):
    if b_i > b_j or a_i > a_j:
        return 0
    return (a_j-a_i) * (b_j-b_i) / (1 - theta * (1 - (a_j-a_i)) * (1 - (b_j-b_i)))


def ali_mikhail_haq_copula_bel(a_j, a_i, b_j, b_i, theta):
    if b_i > b_j or a_i > a_j:
        return 0
    return a_j * b_j / (1 - theta * (1 - a_j) * (1 - b_j)) + a_i * b_i / (1 - theta * (1 - a_i) * (1 - b_i)) \
           - a_i * b_j / (1 - theta * (1 - a_i) * (1 - b_j)) - a_j * b_i / (1 - theta * (1 - a_j) * (1 - b_i))

def clayton_copula(u, v, theta):
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s ; b=%s\\"
                      "Cropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    if theta < -1 or theta == 0:
        err = "Theta should be in [-1, infty[ / {0}: theta=%s" % theta
        raise ValueError(err)
    return max(u**(-theta) + v**(-theta) - 1, 0)**(-1/theta)


def gumbel_copula(u, v, theta):
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s ; b=%s\\"
                      "Cropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    if theta <= 0 or theta > 1:
        err = "Theta should be in ]0, 1]: theta=%s" % theta
        raise ValueError(err)
    return u*v*np.exp(-theta*np.log(u)*np.log(v))


if __name__ == "__main__":
    th = np.arange(-1, 0, 0.1)  # theta in [-1,0(
    xy_res = 0.1
    threshold = 0.00001
    pinf = np.vectorize(ali_mikhail_haq_copula_pinf)
    bel = np.vectorize(ali_mikhail_haq_copula_bel)

    a = b = np.hstack([np.arange(0, 1, xy_res), 1])
    a_j = a[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    a_i = a[:, np.newaxis, np.newaxis, np.newaxis]
    b_j = b[:, np.newaxis, np.newaxis]
    b_i = b[:, np.newaxis]

    bel_arr = bel(a_j, a_i, b_j, b_i, th)  # array of dim 5 and shape=(len(a), len(a), len(b), len(b), len(th))
    pinf_arr = pinf(a_j, a_i, b_j, b_i, th)
    print(bel_arr[np.logical_or(bel_arr < -threshold, bel_arr > 1+threshold)])
    print(np.argwhere((pinf_arr - bel_arr < threshold)))