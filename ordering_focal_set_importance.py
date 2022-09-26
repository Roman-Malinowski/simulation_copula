import types
import numpy as np
import itertools

import pandas as pd
from Copulas import *


def check_possibility_distribution(poss: dict) -> None:
    """
    Check if the possibility distribution is well-defined
    """
    assert 1 in poss.values(), "The possibility distribution must include 1 at least once ! %s" % poss
    for k in poss.keys():
        assert "," not in k, "The keys cannot contain ',' in it. Please change your key names: %s" % k
    return


def mass_from_possibility(poss: dict) -> dict:
    """Find the focal sets and compute their mass from a possibility distribution.
    Return a DataFrame with columns 'Focal sets' and 'mass'
    Example:
      Focal sets  mass
    0   x1,x2,x3  0.2
    1      x1,x3  0.3
    2         x1  0.5
    """
    check_possibility_distribution(poss)

    focal_sets = pd.DataFrame(columns=["Focal sets", "mass"])

    # Create a row for the empty set with null mass for later difference
    focal_sets.loc[0, ["Focal sets", "mass"]] = ["empty", 0]

    # Compute focal sets
    for i, alpha in enumerate(set(poss.values())):
        key_focal = [k for k in poss.keys() if poss[k] >= alpha]
        key_focal = ",".join(key_focal)
        focal_sets.loc[i + 1, ["Focal sets", "mass"]] = [key_focal, alpha]

    # Compute masses
    focal_sets = focal_sets.sort_values(by="mass", axis=0, ascending=True, ignore_index=True)
    focal_sets["mass"] = focal_sets["mass"].diff()

    return focal_sets[focal_sets["mass"] > 0].reset_index(drop=True)


def joint_mass(mass_x: pd.DataFrame, mass_y: pd.DataFrame, order_x: pd.DataFrame, order_y: pd.DataFrame,
               copula: types.FunctionType) -> pd.DataFrame:
    """
    Compute the joint mass from two marginal masses, with a specified order and a specified copula
    mass_x, mass_y: pd.Dataframe with columns 'Focal sets' and 'mass'.
    Example
      Focal sets  mass
    0   x1,x2,x3  0.2
    1      x1,x3  0.3
    2         x1  0.5

    order_x, order_y: pd.DataFrame with column 'order' and index 'Focal sets'.
    Example
                    order
    Focal sets
    x1              2
    x1,x3           1
    x1,x2,x3        3

    Copula: a function that takes as argument two floats between 0 and 1 and returns the copula on those numbers

    return mass_joint: pd.DataFrame. A dataframe with MultiIndex (product of 'Focal sets' from mass_x and mass_y)
    and column 'mass'
    """
    # TODO: Check that the focal sets created by mass_from_possibility are written the same way as order_x and order_y
    multi_index = pd.MultiIndex.from_product([mass_x["Focal sets"], mass_y["Focal sets"]], names=("X", "Y"))
    mass_joint = pd.DataFrame(columns=["mass"], index=multi_index)

    for m_x, m_y in multi_index:
        index = order_x[order_x["order"] < order_x.loc[m_x, "order"]].index
        sum_x_inf = mass_x[mass_x["Focal sets"].isin(index.values)]["mass"].sum(axis=0)
        sum_x_sup = sum_x_inf + float(mass_x[mass_x["Focal sets"] == m_x]["mass"])

        index = order_y[order_y["order"] < order_y.loc[m_y, "order"]].index
        sum_y_inf = mass_y[mass_y["Focal sets"].isin(index.values)]["mass"].sum(axis=0)
        sum_y_sup = sum_y_inf + float(mass_y[mass_y["Focal sets"] == m_y]["mass"])

        mass_joint.loc[(m_x, m_y), "mass"] = copula(sum_x_sup, sum_y_sup) - copula(sum_x_sup, sum_y_inf) - copula(
            sum_x_inf, sum_y_sup) + copula(sum_x_inf, sum_y_inf)

    return mass_joint


def generator_probability(poss: dict, epsilon: float = 1e-6, num=11) -> pd.DataFrame:
    """
    Generator function for probabilities. Yields a sampled probability respecting the probability ranges on atoms

    epsilon: float. Margin of error when verifying that the probability does indeed belong to the credal set
    num: int. The number of point for the np.linspace sampling
    """
    mass = mass_from_possibility(poss)

    prob_range = pd.DataFrame(columns=["Nec", "Pl"], index=pd.Index(poss.keys()).union(pd.Index(mass["Focal sets"])))

    for event in prob_range.index:
        inclusion = [k in event for k in mass["Focal sets"]]
        prob_range.loc[event, "Nec"] = mass[inclusion]["mass"].sum()

        intersection = [len(set(event.split(",")) & set(k.split(","))) > 0 for k in mass["Focal sets"]]
        prob_range.loc[event, "Pl"] = mass[intersection]["mass"].sum()

    # TODO How to do the same with any number of atoms?
    p = pd.DataFrame(columns=["P"], index=pd.Index(poss.keys()), dtype=float)
    for x1 in np.linspace(prob_range.loc[p.index[0], "Nec"], prob_range.loc[p.index[0], "Pl"],
                          num=num):
        for x2 in np.linspace(prob_range.loc[p.index[1], "Nec"], prob_range.loc[p.index[1], "Pl"], num=num):
            p.loc[p.index[0:3], "P"] = [x1, x2, 1 - x1 - x2]

            if np.any(p["P"] < 0):
                continue

            for focal_set in mass["Focal sets"]:
                inclusion = [k in focal_set for k in mass["Focal sets"]]
                nec = mass[inclusion]["mass"].sum()

                intersection = [len(set(focal_set.split(",")) & set(k.split(","))) > 0 for k in mass["Focal sets"]]
                pl = mass[intersection]["mass"].sum()

                # TODO: I believe only comparing to the necessity is enough
                if (pl + epsilon < p.loc[focal_set.split(","), "P"].sum()) | (
                        p.loc[focal_set.split(","), "P"].sum() < nec - epsilon):
                    continue
            yield p


def joint_proba_on_atoms(p_x: pd.DataFrame, p_y: pd.DataFrame, order_x: pd.DataFrame, order_y: pd.DataFrame,
                         copula: types.FunctionType) -> pd.DataFrame:
    """
        Compute the joint probability on atoms from two marginal probabilities,
        with a specified order and a specified copula
        p_x, p_y: pd.Dataframe with columns 'P'.
        Example
            P
        x1  0.5
        x2  0.2
        x3  0.3

        order_x, order_y: pd.DataFrame with column 'order'.
        Example
                order
        x3        3
        x1        1
        x2        2

        Copula: a function that takes as argument two floats between 0 and 1 and returns the copula on those numbers

        return p_xy: pd.DataFrame. A dataframe with MultiIndex (product of indexes from p_x and p_y) and column 'P'
        """
    p_xy = pd.DataFrame(columns=["P"], index=pd.MultiIndex.from_product([p_x.index, p_y.index], names=["X", "Y"]))

    for a_x, a_y in p_xy.index:
        index = order_x[order_x["order"] < order_x.loc[a_x, "order"]].index
        sum_x_inf = p_x.loc[index, "P"].sum(axis=0)
        sum_x_sup = sum_x_inf + float(p_x.loc[a_x, "P"])

        index = order_y[order_y["order"] < order_y.loc[a_y, "order"]].index
        sum_y_inf = p_y.loc[index, "P"].sum(axis=0)
        sum_y_sup = sum_y_inf + float(p_y.loc[a_y, "P"])

        p_xy.loc[(a_x, a_y), "P"] = copula(sum_x_sup, sum_y_sup) - copula(sum_x_sup, sum_y_inf) - copula(
            sum_x_inf, sum_y_sup) + copula(sum_x_inf, sum_y_inf)
    return p_xy


def approximate_robust_credal_set(poss_x: dict, poss_y: dict, order_x_p: pd.DataFrame, order_y_p: pd.DataFrame,
                                  copula: types.FunctionType) -> pd.DataFrame:
    """
    Approximate the robust credal set from marginal credal sets
    poss_x, poss_y: dict. Contains the possibility distribution
    Rows are atoms, columns are "Nec" and "Pl"
    order_x_p, order_y_p: pd.DataFrame. order on atoms ('precise' order). Rows are atoms, column is 'order'
    copula: a function that takes as argument two floats between 0 and 1 and returns the copula on those numbers

    Output robust_df: A DataFrame with multi index and a single column "P_inf" containing the approximation of the lower
    probability on events.

    We generate sampled marginal probabilities from 'prob_range' and compute the joint_proba_on_atoms from them.
    Then we compare that to robust_df and keep the lowest proba for every event.
    """
    full_events_x = []
    for k in range(1, len(poss_x.keys()) + 1):
        full_events_x += [",".join(list(j)) for j in itertools.combinations(poss_x.keys(), k)]

    full_events_y = []
    for k in range(1, len(poss_y.keys()) + 1):
        full_events_y += [",".join(list(j)) for j in itertools.combinations(poss_y.keys(), k)]

    multi = pd.MultiIndex.from_product([full_events_x, full_events_y], names=["X", "Y"])
    robust_df = pd.DataFrame(columns=["P_inf", "P"], index=multi)
    robust_df["P_inf"] = 1

    generator_x = generator_probability(poss_x)
    generator_y = generator_probability(poss_y)

    for p_x in generator_x:
        for p_y in generator_y:
            p_xy = joint_proba_on_atoms(p_x, p_y, order_x_p, order_y_p, copula)
            for x, y in robust_df.index:
                x_i, y_i = x.split(","), y.split(",")
                atoms = list(itertools.product(*[x_i, y_i]))
                if robust_df.loc[(x, y), "P_inf"] > p_xy.loc[atoms, "P"].sum():
                    robust_df.loc[(x, y), "P_inf"] = np.round(p_xy.loc[atoms, "P"].sum(), 6)
                    robust_df.loc[(x, y), "P"] = str(list(p_x["P"].round(6))) + " | " + str(list(p_y["P"].round(6)))
    return robust_df


if __name__ == "__main__":
    # Possibility distributions
    poss_x = {"x1": 1, "x2": 0.2, "x3": 0.5}
    poss_y = {"y1": 0.1, "y2": 1, "y3": 0.35}

    # Orderings
    order_x = pd.DataFrame(columns=["order"], index=pd.Index(["x1", "x1,x3", "x1,x2,x3"], name="Focal sets"))
    order_x["order"] = [1, 2, 3]
    order_x_precise = pd.DataFrame(columns=["order"], index=pd.Index(["x1", "x2", "x3"]))
    order_x_precise["order"] = [1, 2, 3]

    order_y = pd.DataFrame(columns=["order"], index=pd.Index(["y2", "y2,y3", "y1,y2,y3"]))
    order_y["order"] = [2, 3, 1]
    order_y_precise = pd.DataFrame(columns=["order"], index=pd.Index(["y1", "y2", "y3"]))
    order_y_precise["order"] = [1, 2, 3]

    p_rob = approximate_robust_credal_set(poss_x, poss_y, order_x_precise, order_y_precise, min_copula)
    print(p_rob)
    p_rob.to_csv("robust_set.csv")
    # print(joint_mass(mass_from_possibility(poss_x), mass_from_possibility(poss_y), order_x, order_y, min_copula))
