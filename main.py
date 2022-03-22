import types
import pandas as pd
import itertools
import numpy as np
import os.path as op
import warnings


def generate_pi_nec_df(poss_dist: dict) -> pd.DataFrame:
    """
    Create a dataframe containing the possiblity measure and necessity measure computed from a possibility distribution.
     Possibility and necessity measures are columns and events are rows. Do not compute the empty set.
    :param poss_dist: A dictionary representing a possibility distribution. Keys are atoms and values are the value of
    the possibility distribution
    :return: panda dataframe
    """
    for k in poss_dist.keys():
        if "'" in k:
            raise IndexError("Cannot use ',' in the dictionary keys")

    coord = list()
    for k in range(1, len(poss_dist) + 1):
        coord += [list(j) for j in itertools.combinations(poss_dist.keys(), k)]

    possibility = generate_pi_measure(coord, poss_dist)
    necessity = generate_nec_measure(coord, poss_dist)

    return possibility.join(necessity)


def generate_pi_measure(coord: list, poss_dist: dict) -> pd.DataFrame:
    """
    Compute the possibility measure for a list of events and a possibility distribution
    :param coord: a list of events. Should not include the empty set.
    Example [['x_1'], ['x_2'], ['x_3'], ['x_1', 'x_2'], ['x_1', 'x_3'], ['x_2', 'x_3'], ['x_1', 'x_2', 'x_3']]
    :param poss_dist: A dictionary representing a possibility distribution. Keys are atoms and values are the value of
    the possibility distribution
    :return: A panda DataFrame. Indexes are the events of coord, column is "possibility"
    """
    pi_ = dict()
    for k in coord:
        pi_[",".join(k)] = max([poss_dist[a] for a in k])
    return pd.DataFrame(data=pi_.values(), index=pi_.keys(), columns=["possibility"])


def generate_nec_measure(coord: list, poss_dist: dict) -> pd.DataFrame:
    """

    :param coord: a list of events. Should not include the empty set.
    Example [['x_1'], ['x_2'], ['x_3'], ['x_1', 'x_2'], ['x_1', 'x_3'], ['x_2', 'x_3'], ['x_1', 'x_2', 'x_3']]
    :param poss_dist: A dictionary representing a possibility distribution. Keys are atoms and values are the value of
    the possibility distribution
    :return: A panda DataFrame. Indexes are the events of coord, column is "possibility"
    """
    nec_ = dict()
    for k in coord:
        k_c = [a for a in poss_dist.keys() if a not in k]
        if not k_c:
            nec_[",".join(k)] = 1
        else:
            nec_[",".join(k)] = 1 - max([poss_dist[a] for a in k_c])
    return pd.DataFrame(data=nec_.values(), index=nec_.keys(), columns=["necessity"])


def generate_sampled_proba_measures(pos_df: pd.DataFrame, step=0.1) -> pd.DataFrame:
    """
    Generate samples of probability given a possibility/necessity dataframe.
    For each atom of pos_df.index, we compute the range of possible probability values in [necessity, possibility]
    with a given step. We then suppress all the probability distributions that do not add to 1 (with an error of
    step*len(atoms)/100)
    :param pos_df: a possibility/necessity dataframe defined as in generate_pi_nec_df()
    :param step: The step used for sampling probabilities. Have an impact on the error tolerance.
    :return: A panda DataFrame. Each row correspond to a probability mass distribution. Each column correspond to an
    atom.
    """
    # Retrieving only events with one element
    atoms = [k for k in pos_df.index if ',' not in k]
    # Range of probability on those atoms
    ranges = [np.append(np.arange(pos_df.loc[a, "necessity"], pos_df.loc[a, "possibility"], step),
                        [pos_df.loc[a, "possibility"]]) for a in atoms]

    p_tot = np.array(list(itertools.product(*ranges)))

    # Removing proba that do not add to one
    p_tot = p_tot[np.abs(1 - p_tot.sum(axis=1)) <= step * len(atoms) / 100, :]
    return pd.DataFrame(data=p_tot, columns=atoms)


def generate_sampled_cdf(proba_df: pd.DataFrame) -> pd.DataFrame:
    """
    CAREFUL: The order of elements for cumulative distribution will be the order of columns. If created from a
    dictionary, it will be alphabetical order.
    Compute the cumulated distribution function (CDF) from a dataframe containing probability mass distributions.
    :param proba_df: a dataframe containing probability mass distributions defined as in generate_sampled_proba_measures()
    :return: A panda dataframe. Each row correspond to a CDF. Columns are cumulated events.
    Example: pd.DataFrame(data=[[0., 0.5, .1], [0.2, 0.4, 1.]], columns=[["x_1"], ["x_1,x_2"], ["x_1,x_2,x_3"]])
    """
    atoms = list(proba_df.columns)
    if "Index X" in atoms:
        atoms.remove("Index X")
    if "Index Y" in atoms:
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


def expand_df(proba_x_: pd.DataFrame, proba_y_: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    In order to have every combination possible for rows of proba_x_ and proba_y_, it is necessary to duplicate each
    rows. Example:
    proba_x_ = [[a], [b], [c]], columns=["x"]
    proba_y_ = [[e], [f]], columns=["y"]
    it returns
    proba_x_expanded = [[a, 0], [a, 0], [b, 1], [b, 1], [c, 1], [c, 1]], columns=["x", "Index X"]
    proba_y_expanded = [[e, 0], [f, 1], [e, 0], [f, 1], [e, 0], [f, 1]], columns=["y", "Index Y"]
    :param proba_x_: A dataframe with n rows and k columns
    :param proba_y_: A dataframe with m rows and j columns
    :return: two dataframes with n*m rows and k+1 and j+1 columns respectively. A column containing the original index
    is added
    """
    index_x, index_y = np.array(proba_x_.index), np.array(proba_y_.index)

    if "Index X" in proba_x_.columns:
        raise KeyError("'Index X' cannot be the name of a column in the probability mass samples")
    if "Index Y" in proba_y_.columns:
        raise KeyError("'Index Y' cannot be the name of a column in the probability mass samples")

    proba_x_.loc[:, "Index X"] = index_x  # Adds it as a last column
    proba_y_.loc[:, "Index Y"] = index_y

    x_array = proba_x_.to_numpy().tolist()
    y_array = proba_y_.to_numpy().tolist()

    # Cartesian product of the array
    xy_product = list(itertools.product(x_array, y_array))

    proba_x_expanded = pd.DataFrame(data=[k[0] for k in xy_product], columns=proba_x_.columns)
    proba_y_expanded = pd.DataFrame(data=[k[1] for k in xy_product], columns=proba_y_.columns)

    return proba_x_expanded, proba_y_expanded


def generate_joint_proba_measures(proba_x_: pd.DataFrame, proba_y_: pd.DataFrame, copula: types.FunctionType) -> pd.DataFrame:
    """
    Compute the joint probabilities from two marginal probabilities under a given copula. Using the formula:
    P_{XY}(x_i, y_i) = C(F_X(x_i), F_Y(y_i)) + C(F_X(x_{i-1}), F_Y(y_{i-1}))
    - C(F_X(x_i), F_Y(y_{i-1})) - C(F_X(x_{i-1}), F_Y(y_i))
    :param proba_x_: a panda DataFrame representing probability mass distributions (rows) as given by generate_sampled_proba_measures()
    :param proba_y_: a panda DataFrame representing probability mass distributions (rows) as given by generate_sampled_proba_measures()
    :param copula: a python Function taking two arguments (float) and returning the copula computed on those floats
    :return: a panda DataFrame p_xy. Each row corresponds to a joint probability mass distribution. The index of p_xy
    is a pd.MultiIndex (first one corresponds to the index of the X proba in proba_x_, second corresponds to the Y
    proba in the proba_y_)
    p_xy columns (pd.MultiIndex) are the cartesian products of events.
    CAREFUL: proba_x_ and proba_y_ columns correspond to atoms. But for the joint probability, we compute the mass over
    all events and not just atoms (except for the empty set).
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

    cdf_x = generate_sampled_cdf(proba_x_expanded)
    cdf_y = generate_sampled_cdf(proba_y_expanded)

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
    for k_x in range(1, len(atoms_x)+1):
        for k_y in range(1, len(atoms_y)+1):
            if k_x == k_y == 1:
                continue
            coord += [e for e in itertools.product(itertools.combinations(atoms_x, k_x),
                                                   itertools.combinations(atoms_y, k_y))]

    # Computing probabilities for those new elements
    for x_, y_ in coord:
        p_xy.loc[:, (",".join(x_), ",".join(y_))] = p_xy.loc[:, (x_, y_)].sum(axis=1)

    return p_xy


def generate_joint_necessity_with_sklar(possibility_df_x: pd.DataFrame, possibility_df_y: pd.DataFrame,
                                        copula: types.FunctionType) -> pd.DataFrame:
    """
    Compute the joint necessity using Sklar's theorem applied to necessity measures. Nec_{XY} = C(Nec_X, Nec_Y)
    :param possibility_df_x: a possibility/necessity dataframe defined as in generate_pi_nec_df()
    :param possibility_df_y: a possibility/necessity dataframe defined as in generate_pi_nec_df()
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
    Creates a DataFrame for comparing the sampled joint probabilities and the joint necessity
    :param joint_necessity: pd.DataFrame. A DataFrame containing all joint necessities. Same structure as the one
    obtained with generate_joint_necessity_with_sklar()
    :param joint_proba: pd.DataFrame. A DataFrame containing all the samples probabilities. Same structure as the one
    obtained with generate_joint_proba_measures()
    :return: pd.DataFrame. A DataFrame whose columns are a pd.MultiIndex obtain from the product of all the events
    on X and Y. Its rows are "Nec", "Min", "Argmin", "P_X atoms", "P_Y atoms".
    "Nec": the necessity for the event
    "Min": the minimum value of the sampled joint probabilities for the event
    "Argmin": the MultiIndex value for the minimal sampled probability P_min for the event
    "P_X atoms" contains a tuple with the marginal of P_min over the atoms of X (to find manually the results)
    "P_Y atoms" contains a tuple with the marginal of P_min over the atoms of Y (to find manually the results)
    """
    atoms_x = np.unique([k[0] for k in joint_proba.columns if ',' not in k[0]])
    atoms_y = np.unique([k[1] for k in joint_proba.columns if ',' not in k[1]])

    df_compare = pd.DataFrame(columns=joint_proba.columns, index=["Nec", "Min", "Argmin", "P_X atoms", "P_Y atoms"])
    for col in df_compare.columns:
        if joint_proba.shape[0] == 1:
            # loc[:, col] returns a float, not a Dataframe. squeeze() will generate AttributeError
            idx_min = joint_proba.index[0]
            min_val = joint_proba.loc[:, col].min(axis=0)
        else:
            idx_min = joint_proba.loc[:, col].squeeze().idxmin(axis=0)  # Only keeps the first occurrence
            min_val = joint_proba.loc[:, col].min(axis=0)
        df_compare.loc[("Min", "Argmin"), col] = np.array([min_val, idx_min], dtype=object)

        df_compare.loc["Nec", col] = joint_necessity.loc[0, col]

        # As C(u,1) = u and C(1,v), we can retrieve marginals easily this way.
        # But there will be more computation errors!
        df_compare.loc["P_X atoms", col] = tuple([joint_proba.loc[idx_min, (k, ",".join(atoms_y))] for k in atoms_x])
        df_compare.loc["P_Y atoms", col] = tuple([joint_proba.loc[idx_min, (",".join(atoms_x), k)] for k in atoms_y])

    return df_compare


def create_differences_df(poss_distrib_x: dict, poss_distrib_y: dict, copula: types.FunctionType)\
        -> (bool, pd.DataFrame):
    pos_df_x_ = generate_pi_nec_df(poss_distrib_x)
    proba_x_ = generate_sampled_proba_measures(pos_df_x_)

    pos_df_y_ = generate_pi_nec_df(poss_distrib_y)
    proba_y_ = generate_sampled_proba_measures(pos_df_y_)

    proba_join_ = generate_joint_proba_measures(proba_x_, proba_y_, copula)
    proba_join_ = proba_join_.round(decimals=10)
    nec_sklar = generate_joint_necessity_with_sklar(pos_df_x_, pos_df_y_, copula)
    nec_sklar = nec_sklar.round(decimals=10)

    df = compare_robust_to_sklar(nec_sklar, proba_join_)

    return df.loc[:, df.loc["Min", :] > df.loc["Nec", :] + float(1e-5)].empty, df


def sample_possibility_distribution(set_values: list, range_of_values: list) -> list:
    """
    Create a list of possible possibility distributions given a set and acceptable values
    of the possibility distribution.
    :param set_values: list of strings. Example: set_values = ["x_1", "x_2", "x_3"]
    :param range_of_values: list of floats between 0 and 1. Example [0.0, 0.1, 0.5, 0.9]
    :return: list of tuple. Each tuple has len(set_values) elements corresponding to the value of the possibility
    distribution for the element.
    """
    n_ = len(set_values)
    list_of_pi_ = list()
    for i in range(1, n_ + 1):
        ones_ = list(itertools.combinations(set_values, i))
        for one_ in ones_:
            range_ = []  # Will contain all the possible values for each x_i
            for x in set_values:
                if x in one_:
                    range_ += [[1.]]
                else:
                    range_ += [range_of_values]
            # Cartesian product of all possible values
            list_of_pi_ += list(itertools.product(*range_))

    return list_of_pi_


def sample_joint_possibilities_distributions(max_xy: int, copula: types.FunctionType, folder: str= "", step: float=0.1) -> None:
    # Creating all possibility distributions possible
    save_number = 0
    decimal_to_round = int(-np.floor(np.log10(step)))
    range_of_values = list(np.around(np.arange(0, 1, step), decimal_to_round))  # np.arange creates small errors

    # Changing the number of elements of X
    i = 0
    pass_to_next_set = False
    for n_x in range(3, max_xy + 1):
        X = ["x_%s" % i for i in range(1, n_x + 1)]
        list_of_pi_x = sample_possibility_distribution(X, range_of_values)

        # Doing the same for Y
        for n_y in range(n_x, max_xy + 1):
            if n_y == 3:
                continue
            i += 1
            print(i, "\n")
            Y = ["y_%s" % i for i in range(1, n_y + 1)]
            list_of_pi_y = sample_possibility_distribution(Y, range_of_values)

            for i_x, p_x in enumerate(list_of_pi_x):
                poss_distib_x = dict(zip(X, p_x))

                for i_y, p_y in enumerate(list_of_pi_y):
                    print("\r%s / %s" % (i_x*len(list_of_pi_y)+i_y, len(list_of_pi_y)*len(list_of_pi_x)), end="")
                    poss_distib_y = dict(zip(Y, p_y))

                    empty, df = create_differences_df(poss_distib_x, poss_distib_y, copula)
                    if not empty:
                        df.to_csv(op.join(folder, "df_%s.csv" % save_number))
                        print("Saved df_%s.csv\n" % save_number)
                        save_number += 1
                        pass_to_next_set = True
                        break
                if pass_to_next_set:
                    pass_to_next_set = False
                    break

    return


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    sample_joint_possibilities_distributions(5, min_copula, "csv_files", step=0.2)
