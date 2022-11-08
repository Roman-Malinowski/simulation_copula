import warnings
import numpy as np


def min_copula(u: float, v: float) -> float:
    """
    The minimum copula
    :param u: a float between 0 and 1
    :param v: a float between 0 and 1
    :return: a float between 0 and 1 corresponding to C(u,v)
    """
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s | v=%s\nCropping the values." % (u, v), UserWarning)
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
        warnings.warn("u and v should be between 1 and 0. u=%s | v=%s\nCropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    return max(0., u + v - 1)


def ali_mikhail_haq_copula(u: float, v: float, theta: float):
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s | v=%s\nCropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    if theta < -1 or theta >= 1:
        err = "Theta should be in [-1, 1[: theta=%s" % theta
        raise ValueError(err)
    if u==0. or v==0.:
        return 0.
    else:
        return u * v / (1 - theta * (1 - u) * (1 - v))


def clayton_copula(u: float, v: float, theta: float):
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s | v=%s\nCropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    if theta < -1 or theta == 0:
        err = "Theta should be in [-1, infty[ / {0}: theta=%s" % theta
        raise ValueError(err)
    if u==0. or v==0.:
        return 0.
    else:
        return max(u**(-theta) + v**(-theta) - 1, 0)**(-1/theta)


def gumbel_copula(u: float, v: float, theta: float):
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s | v=%s\nCropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    if theta <= 0 or theta > 1:
        err = "Theta should be in ]0, 1]: theta=%s" % theta
        raise ValueError(err)
    if u==0. or v==0.:
        return 0.
    else:
        return u*v*np.exp(-theta*np.log(u)*np.log(v))


def product_copula(u: float, v: float):
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s | v=%s\nCropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    return u*v


def frank_copula(u: float, v: float, theta: float):
    if 0. > u or 1. < u or 0 > v or 1 < v:
        warnings.warn("u and v should be between 1 and 0. u=%s | v=%s\nCropping the values." % (u, v), UserWarning)
        u = max(0., u)
        u = min(1., u)
        v = max(0., v)
        v = min(1., v)
    if theta == 0:
        err = "Theta should not be equal to 0"
        raise ValueError(err)
    if u==0. or v==0.:
        return 0.
    else:
        return -np.log(1 + (np.exp(-theta*u) - 1) * (np.exp(-theta*v) - 1) / (np.exp(-theta) - 1)) / theta 
