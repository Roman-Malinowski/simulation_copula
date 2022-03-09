import types
import pandas as pd
import itertools
import numpy as np


def possibility_df(poss_dist: dict) -> pd.DataFrame:
    """
    Create a dataframe containing the possiblity measure and necessity measure computed from a possibility distribution.
     Possibility and necessity measures are columns and events are rows. Do not compute the empty set and the full set.
    :param poss_dist: A dictionary representing a possibility distribution. Keys are atoms and values are the value of
    the possibility distribution
    :return: panda dataframe
    """
    for k in poss_dist.keys():
        if "'" in k:
            raise IndexError("Cannot use ',' in the dictionary keys")

    coord = list()
    for k in range(1, len(poss_dist)):
        coord += [list(j) for j in itertools.combinations(poss_dist.keys(), k)]

    possibility = pi_measure(coord, poss_dist)
    necessity = nec_measure(coord, poss_dist)

    return possibility.join(necessity)


def pi_measure(coord: list, poss_dist: dict) -> pd.DataFrame:
    """
    Compute the possibility measure for a list of events and a possibility distribution
    :param coord: a list of events. Should not include the empty set.
    Example [['x_1'], ['x_2'], ['x_3'], ['x_1', 'x_2'], ['x_1', 'x_3'], ['x_2', 'x_3']]
    :param poss_dist: A dictionary representing a possibility distribution. Keys are atoms and values are the value of
    the possibility distribution
    :return: A panda DataFrame. Indexes are the events of coord, column is "possibility"
    """
    pi_ = dict()
    for k in coord:
        pi_[",".join(k)] = max([poss_dist[a] for a in k])
    return pd.DataFrame(data=pi_.values(), index=pi_.keys(), columns=["possibility"])


def nec_measure(coord: list, poss_dist: dict) -> pd.DataFrame:
    """

    :param coord: a list of events. Should not include the empty set.
    Example [['x_1'], ['x_2'], ['x_3'], ['x_1', 'x_2'], ['x_1', 'x_3'], ['x_2', 'x_3']]
    :param poss_dist: A dictionary representing a possibility distribution. Keys are atoms and values are the value of
    the possibility distribution
    :return: A panda DataFrame. Indexes are the events of coord, column is "possibility"
    """
    nec_ = dict()
    for k in coord:
        k_c = [a for a in poss_dist.keys() if a not in k]
        nec_[",".join(k)] = 1 - max([poss_dist[a] for a in k_c])
    return pd.DataFrame(data=nec_.values(), index=nec_.keys(), columns=["necessity"])


def generate_proba(pos_df: pd.DataFrame, step=0.1) -> pd.DataFrame:
    """
    Generate samples of probability given a possibility/necessity dataframe.
    For each atom of pos_df.index, we compute the range of possible probability values in [necessity, possibility]
    with a given step. We then suppress all the probability distributions that do not add to 1 (with an error of
    step*len(atoms)/100)
    :param pos_df: a possibility/necessity dataframe defined as in possibility_df()
    :param step: The step used for sampling probabilities. Have an impact on the error tolerance.
    :return: A panda DataFrame. Each row correspond to a probability mass distribution. Each column correspond to an
    atom.
    """
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
    """
    Compute the cumulated distribution function (CDF) from a dataframe containing probability mass distributions.
    :param proba_df: a dataframe containing probability mass distributions defined as in generate_proba()
    :return: A panda dataframe. Each row correspond to a CDF. Columns are cumulated events.
    Example: pd.DataFrame(data=[[0., 0.5, .1], [0.2, 0.4, 1.]], columns=[["x_1"], ["x_1,x_2"], ["x_1,x_2,x_2"]])
    """
    atoms = list(proba_df.columns)
    if "Index X" in atoms:
        print("Ignoring 'Index X' in CDF computation")
        atoms.remove("Index X")
    if "Index Y" in atoms:
        print("Ignoring 'Index Y' in CDF computation")
        atoms.remove("Index Y")

    cdf_columns = [",".join(atoms[:k + 1]) for k in range(len(atoms))]
    cdf_ = proba_df.loc[:, atoms].cumsum(axis=1)
    cdf_.columns = cdf_columns
    return cdf_


def min_copula(u: float, v: float) -> float:
    """
    The minimun copula
    :param u: a float between 0 and 1
    :param v: a float between 0 and 1
    :return: a float between 0 and 1 corresponding to C(u,v)
    """
    if 0. > u or 1. < u or 0 > v or 1 < v:
        raise ValueError("u and v should be between 1 and 0. u=%s ; b=%s" % (u, v))
    return min(u, v)


def lukaciewicz(u: float, v: float) -> float:
    """
    The lukaciewicz copula
    :param u: a float between 0 and 1
    :param v: a float between 0 and 1
    :return: a float between 0 and 1 corresponding to C(u,v)
    """
    if 0. > u or 1. < u or 0 > v or 1 < v:
        raise ValueError("u and v should be between 1 and 0. u=%s ; b=%s" % (u, v))
    return max(0., u + v - 1)


def expand_df(proba_x_: pd.DataFrame, proba_y_: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    In order to have every combination possible for rows of proba_x_ and proba_y_, it is necessary to duplicate each
    rows. Example:
    proba_x_ = [[a], [b], [c]]
    proba_y_ = [[e], [f]]
    it returns
    proba_x_expanded = [[a], [a], [b], [b], [c], [c]]
    proba_y_expanded = [[e], [e], [f], [f]]
    :param proba_x_: A dataframe with n rows and k columns
    :param proba_y_: A dataframe with m rows and j columns
    :return: two dataframes with n*m rows and k+1 and j+1 columns respectively. A column containing the original index
    is added
    """
    n_x, n_y = proba_x_.shape[0], proba_y_.shape[0]
    index_x, index_y = np.array(proba_x_.index), np.array(proba_y_.index)

    if "Index X" in proba_x_.columns:
        raise KeyError("'Index X' cannot be the name of a column in the probability mass samples")
    if "Index Y" in proba_y_.columns:
        raise KeyError("'Index Y' cannot be the name of a column in the probability mass samples")

    proba_x_.loc[:, "Index X"] = index_x
    proba_x_expanded = proba_x_.set_index(index_x * n_y)

    proba_y_.loc[:, "Index Y"] = index_y
    proba_y_expanded = proba_y_.copy()

    for k in range(1, n_y):
        proba_x_expanded = pd.concat([proba_x_expanded, proba_x_.set_index(index_x*n_y + k)])
    proba_x_expanded.sort_index(inplace=True)

    for j in range(1, n_x):
        proba_y_expanded = pd.concat([proba_y_expanded, proba_y_])
    proba_y_expanded.reset_index(inplace=True, drop=True)

    return proba_x_expanded, proba_y_expanded


def generate_joint_proba(proba_x_: pd.DataFrame, proba_y_: pd.DataFrame, copula: types.FunctionType) -> pd.DataFrame:
    """
    Compute the joint probabilities from two marginal probabilities under a given copula. Using the formula:
    P_{XY}(x_i, y_i) = C(F_X(x_i), F_Y(y_i)) + C(F_X(x_{i-1}), F_Y(y_{i-1}))
    - C(F_X(x_i), F_Y(y_{i-1})) - C(F_X(x_{i-1}), F_Y(y_i))
    :param proba_x_: a panda DataFrame representing probability mass distributions (rows) as given by generate_proba()
    :param proba_y_: a panda DataFrame representing probability mass distributions (rows) as given by generate_proba()
    :param copula: a python Function taking two arguments (float) and returning the copula computed on those floats
    :return: a panda DataFrame p_xy. Each row corresponds to a joint probability mass distribution. The index of p_xy
    is a pd.MultiIndex (first one corresponds to the index of the X proba in proba_x_, second corresponds to the Y
    proba in the proba_y_)
    p_xy columns (pd.MultiIndex) are the cartesian products of events.
    CAREFUL: proba_x_ and proba_y_ columns correspond to atoms. But for the joint probability, we compute the mass over
    all events and not just atoms (except for the empty set and cartesian products containing one of the full set of
    the marginals).
    """
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

    # Expand the lines of the proba to make it more handy to compute every combinations
    proba_x_expanded, proba_y_expanded = expand_df(proba_x_, proba_y_)

    cdf_x = generate_cdf(proba_x_expanded)
    cdf_y = generate_cdf(proba_y_expanded)

    # Adding an empty column so that the index before "x_1" is defined
    cdf_x.loc[:, ""] = 0.
    cdf_y.loc[:, ""] = 0.

    p_xy = pd.DataFrame(columns=indexes)

    for x_, y_ in indexes:
        x_inf, x_sup = x_events[x_]
        y_inf, y_sup = y_events[y_]
        p_xy.loc[:, (x_, y_)] = cdf_x.loc[:, x_sup].combine(cdf_y.loc[:, y_sup], copula) + \
                            cdf_x.loc[:, x_inf].combine(cdf_y.loc[:, y_inf], copula) - \
                            cdf_x.loc[:, x_inf].combine(cdf_y.loc[:, y_sup], copula) - \
                            cdf_x.loc[:, x_sup].combine(cdf_y.loc[:, y_inf], copula)

    # Setting the index to be able to find where each joint proba comes from
    p_xy.set_index(pd.MultiIndex.from_arrays([proba_x_expanded["Index X"], proba_y_expanded["Index Y"]]), inplace=True)

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


def compute_joint_with_sklar(possibility_df_x: pd.DataFrame, possibility_df_y: pd.DataFrame,
                             copula: types.FunctionType) -> pd.DataFrame:
    """
    Compute the joint necessity using Sklar's theorem applied to necessity measures. Nec_{XY} = C(Nec_X, Nec_Y)
    :param possibility_df_x: a possibility/necessity dataframe defined as in possibility_df()
    :param possibility_df_y: a possibility/necessity dataframe defined as in possibility_df()
    :param copula: a python Function taking two arguments (float) and returning the copula computed on those floats
    :return: a panda DataFrame whose columns are the cartesian product of
    possibility_df_x.index and possibility_df_y.index. The DataFrame only has one row, corresponding to the value of
    the joint necessity measure
    """
    index_joint = pd.MultiIndex.from_product([possibility_df_x.index, possibility_df_y.index], names=["X", "Y"])
    joint_df = pd.DataFrame(columns=index_joint)

    for x_, y_ in index_joint:
        joint_df.loc[0, (x_, y_)] = copula(possibility_df_x.loc[x_, "necessity"], possibility_df_y.loc[y_, "necessity"])
    return joint_df


def compare_robust_to_sklar(joint_necessity: pd.DataFrame, joint_proba: pd.DataFrame) -> pd.DataFrame:
    """

    :param joint_necessity:
    :param joint_proba:
    :return:
    """
    min_on_events = joint_proba.min(axis=0)


if __name__ == '__main__':
    possibility_distribution_x = dict(x_1=0.5, x_2=1, x_3=0.3)
    pos_df_x = possibility_df(possibility_distribution_x)
    proba_x = generate_proba(pos_df_x)

    possibility_distribution_y = dict(y_1=1, y_2=0.2, y_3=0.6)
    pos_df_y = possibility_df(possibility_distribution_y)
    proba_y = generate_proba(pos_df_y)

    """proba_x = pd.DataFrame(data=np.array([[0.1, 0.7, 0.2], [0.5, 0.5, 0.]]), columns=["x_1", "x_2", "x_3"])
    proba_y = pd.DataFrame(data=np.array([[0.2, 0.8], [0.6, 0.4], [0.0, 1]]), columns=["y_1", "y_2"])"""
    proba_join = generate_joint_proba(proba_x, proba_y, min_copula)
    proba_join = proba_join.round(decimals=10)
    proba_join.to_csv("test.csv")

    print(proba_join)
