import types
import pandas as pd
import itertools
import numpy as np


def possibility_set(poss_dist: dict) -> pd.DataFrame:
    for k in poss_dist.keys():
        if "'" in k:
            raise IndexError("Cannot use ',' in the dictionary keys")

    coord = list()
    for k in range(1, len(poss_dist)):
        coord += [list(j) for j in itertools.combinations(poss_dist.keys(), k)]

    # coord : [['x_1'], ['x_2'], ['x_3'], ['x_1', 'x_2'], ['x_1', 'x_3'], ['x_2', 'x_3']]
    possibility = pi_measure(coord, poss_dist)
    necessity = nec_measure(coord, poss_dist)

    return possibility.join(necessity)


def pi_measure(coord: list, poss_dist: dict) -> pd.DataFrame:
    pi_ = dict()
    for k in coord:
        pi_[",".join(k)] = max([poss_dist[a] for a in k])
    return pd.DataFrame(data=pi_.values(), index=pi_.keys(), columns=["possibility"])


def nec_measure(coord: list, poss_dist: dict) -> pd.DataFrame:
    nec_ = dict()
    for k in coord:
        k_c = [a for a in poss_dist.keys() if a not in k]
        nec_[",".join(k)] = 1 - max([poss_dist[a] for a in k_c])
    return pd.DataFrame(data=nec_.values(), index=nec_.keys(), columns=["necessity"])


def generate_proba(pos_df: pd.DataFrame, step=0.1) -> pd.DataFrame:
    # Retrieving only events with one element
    atoms = [k for k in pos_df.index if ',' not in k]
    # Range of probability on those atoms
    ranges = [np.append(np.arange(pos_df.loc[a, "necessity"], pos_df.loc[a, "possibility"], step),
                        [pos_df.loc[a, "possibility"]]) for a in atoms]
    n_lines = np.prod([len(x) for x in ranges])
    p_tot = np.zeros((n_lines, len(atoms)))

    n_occur = n_lines
    n_period = 1
    for k in range(len(atoms)):
        n_occur /= len(ranges[k])
        p_tot[:, k] = np.array([[i] * int(n_occur) for i in ranges[k]] * int(n_period)).flatten()
        n_period *= len(ranges[k])

    # Removing proba that do not add to one
    p_tot = p_tot[np.abs(1 - p_tot.sum(axis=1)) <= step * len(atoms) / 100, :]
    return pd.DataFrame(data=p_tot, columns=atoms)


def generate_cdf(proba_df: pd.DataFrame) -> pd.DataFrame:
    atoms = list(proba_df.columns)
    cdf_columns = [",".join(atoms[:k + 1]) for k in range(len(atoms))]
    cdf_ = proba_df.cumsum(axis=1)
    cdf_.columns = cdf_columns
    return cdf_


def min_copula(u: float, v: float) -> float:
    if 0. > u or 1. < u or 0 > v or 1 < v:
        raise ValueError("u and v should be between 1 and 0. u=%s ; b=%s" % (u, v))
    return min(u, v)


def lukaciewicz(u: float, v: float) -> float:
    if 0. > u or 1. < u or 0 > v or 1 < v:
        raise ValueError("u and v should be between 1 and 0. u=%s ; b=%s" % (u, v))
    return max(0, u + v - 1)


def generate_joint_proba(proba_x_: pd.DataFrame, proba_y_: pd.DataFrame, copula: types.FunctionType) -> pd.DataFrame:
    atoms_x = list(proba_x_.columns)
    atoms_y = list(proba_y_.columns)
    indexes = pd.MultiIndex.from_product([atoms_x, atoms_y], names=["X", "Y"])

    # Creating a dictionary containing the cumulate event right before and containing each element
    x_events = {}
    for i, x_ in enumerate(atoms_x):
        x_events[x_] = (",".join(atoms_x[:i]), ",".join(atoms_x[:i + 1]))

    y_events = {}
    for i, y_ in enumerate(atoms_y):
        y_events[y_] = (",".join(atoms_y[:i]), ",".join(atoms_y[:i + 1]))

    cdf_x = generate_cdf(proba_x_)
    cdf_y = generate_cdf(proba_y_)
    cdf_x.loc[:, ""] = 0.
    cdf_y.loc[:, ""] = 0.

    p_xy = pd.DataFrame(columns=indexes)
    print(p_xy)
    for x_, y_ in indexes:
        x_inf, x_sup = x_events[x_]
        y_inf, y_sup = y_events[y_]
        p_xy.loc[:, (x_, y_)] = cdf_x.loc[:, x_sup].combine(cdf_y.loc[:, y_sup], copula) + \
                            cdf_x.loc[:, x_inf].combine(cdf_y.loc[:, y_inf], copula) - \
                            cdf_x.loc[:, x_inf].combine(cdf_y.loc[:, y_sup], copula) - \
                            cdf_x.loc[:, x_sup].combine(cdf_y.loc[:, y_inf], copula)

    # Computing elements that are combinations of 2D-atoms
    coord = list()
    for k_x in range(1, len(atoms_x)):
        for k_y in range(1, len(atoms_y)):
            if k_x == k_y == 1:
                continue
            coord += [e for e in itertools.product(itertools.combinations(atoms_x, k_x),
                                                   itertools.combinations(atoms_y, k_y))]

    # Computing probabilities for those new elements
    for x_, y_ in coord:
        p_xy.loc[:, (",".join(x_), ",".join(y_))] = p_xy.loc[:, (x_, y_)].sum(axis=1)

    return p_xy


if __name__ == '__main__':
    # TODO What if we don't sample the same number of probabilities ?
    possibility_distribution_x = dict(x_1=0.5, x_2=1, x_3=0.3)
    pos_set_x = possibility_set(possibility_distribution_x)
    proba_x = generate_proba(pos_set_x)

    possibility_distribution_y = dict(y_1=1, y_2=0.2, y_3=0.6)
    pos_set_y = possibility_set(possibility_distribution_y)
    proba_y = generate_proba(pos_set_y)

    proba_join = generate_joint_proba(proba_x, proba_y, min_copula)
    print(proba_join.loc[7, :])
