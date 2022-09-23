import types

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

    return mass_joint: pd.DataFrame. A dataframe with MultiIndex (product of 'Focal sets' from mass_x and mass_y) and column 'mass'
    """
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


if __name__ == "__main__":
    poss_x = {"x1": 1, "x2": 0.2, "x3": 0.5}
    poss_y = {"y1": 0.1, "y2": 1, "y3": 0.35}

    order_x = pd.DataFrame(columns=["order"], index=pd.Index(["x1", "x1,x3", "x1,x2,x3"], name="Focal sets"))
    order_x["order"] = [1, 2, 3]

    order_y = pd.DataFrame(columns=["order"], index=pd.Index(["y2", "y2,y3", "y1,y2,y3"], name="Focal sets"))
    order_y["order"] = [2, 3, 1]

    print(joint_mass(mass_from_possibility(poss_x), mass_from_possibility(poss_y), order_x, order_y, min_copula))
