import types
import numpy as np
import itertools

import pandas as pd
from Copulas import *


def check_possibility_distribution(poss: dict) -> None:
    """
    Check if the possibility distribution is well defined
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


def probability_range_from_poss(poss: dict) -> pd.DataFrame:
    """
    Create a DataFrame with probability ranges for atoms, with columns being Necessity and Plausibility
    poss: A dictionary with atoms being the keys and possibility the values
    return p: pd.DataFrame. Columns=["Nec", "Pl"], rows are atoms
    Example
                    Nec     Pl
    x1              0.5     1
    x2              0       0.2
    x3              0       0.5
    """
    mass = mass_from_possibility(poss)

    p = pd.DataFrame(columns=["Nec", "Pl"], index=pd.Index(poss.keys()))

    for atom in p.index:
        # Kind of useless as there will always be only one atom to be included
        inclusion = [k in atom for k in mass["Focal sets"]]
        p.loc[atom, "Nec"] = mass[inclusion]["mass"].sum()

        intersection = [atom in k for k in mass["Focal sets"]]
        p.loc[atom, "Pl"] = mass[intersection]["mass"].sum()
    return p


def generator_probability(prob_range: pd.DataFrame) -> pd.DataFrame:
    """
    Generator function for probabilities. Yields a sampled probability respecting the probability ranges on atoms
    prob_range: pd.DataFrame.  Columns=["Nec", "Pl"]. 3 Rows.  Obtained from probability_range_from_poss
    Example
                    Nec     Pl
    x1              0.5     1
    x2              0       0.2
    x3              0       0.5
    """

    # TODO How to do the same with any number of atoms?
    p = pd.DataFrame(columns=["P"], index=prob_range.index)
    for x1 in np.linspace(prob_range.loc[prob_range.index[0], "Nec"], prob_range.loc[prob_range.index[0], "Pl"],
                          num=11):
        pl_2 = min(1 - x1, prob_range.loc[prob_range.index[1], "Pl"])
        if pl_2 < prob_range.loc[prob_range.index[1], "Nec"]:
            p["P"] = [x1, 0, 0]
            yield p
        else:
            for x2 in np.linspace(prob_range.loc[prob_range.index[1], "Nec"], pl_2, num=11):
                pl_3 = min(1 - x1 - x2, prob_range.loc[prob_range.index[2], "Pl"])
                if pl_3 < prob_range.loc[prob_range.index[2], "Nec"]:
                    p["P"] = [x1, x2, 0]
                    yield p
                else:
                    for x3 in np.linspace(prob_range.loc[prob_range.index[2], "Nec"], pl_3, num=11):
                        p["P"] = [x1, x2, x3]
                        yield p
    yield None


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
        sum_x_inf = p_x[index, "P"].sum(axis=0)
        sum_x_sup = sum_x_inf + float(p_x[a_x, "P"])

        index = order_y[order_y["order"] < order_y.loc[a_y, "order"]].index
        sum_y_inf = p_y[index, "P"].sum(axis=0)
        sum_y_sup = sum_y_inf + float(p_y[a_y, "P"])

        p_xy.loc[(a_x, a_y), "P"] = copula(sum_x_sup, sum_y_sup) - copula(sum_x_sup, sum_y_inf) - copula(
            sum_x_inf, sum_y_sup) + copula(sum_x_inf, sum_y_inf)
    return p_xy


def approximate_robust_credal_set(prob_range_x: pd.DataFrame, prob_range_y: pd.DataFrame,
                                  order_x_p, order_y_p, copula: types.FunctionType) -> pd.DataFrame:
    full_events_x = []
    for k in range(1, len(prob_range_x.index) + 1):
        full_events_x += [",".join(list(j)) for j in itertools.combinations(prob_range_x.index, k)]

    full_events_y = []
    for k in range(1, len(prob_range_y.index) + 1):
        full_events_y += [",".join(list(j)) for j in itertools.combinations(prob_range_y.index, k)]

    multi = pd.MultiIndex.from_product([full_events_x, full_events_y], names=["X", "Y"])
    robust_df = pd.DataFrame(columns=["P_inf"], index=multi)
    robust_df["P_inf"] = 1

    generator_x = generator_probability(prob_range_x)
    generator_y = generator_probability(prob_range_y)

    for p_x in generator_x:
        for p_y in generator_y:
            p_xy = joint_proba_on_atoms(p_x, p_y, order_x_p, order_y_p, copula)
            for x, y in robust_df.index:
                x_i, y_i = x.split(","), y.split(",")
                unique_combinations = []
                # TODO case if len(x_i)< len(y_i)
                permut = itertools.permutations(x_i, len(y_i))
                for comb in permut:
                    zipped = zip(comb, y_i)
                    unique_combinations += list(zipped)

                robust_df[(x, y), "P"] = min(robust_df[(x, y), "P"], p_xy.loc[unique_combinations, "P"].sum())

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

    print(probability_range_from_poss(poss_x))
    # print(joint_mass(mass_from_possibility(poss_x), mass_from_possibility(poss_y), order_x, order_y, min_copula))
