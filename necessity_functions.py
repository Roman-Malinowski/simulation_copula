import typing

import pandas as pd


class NecessityUnivariate:

    def __init__(self, poss: dict, order_focal_sets: typing.Optional[pd.DataFrame] = None) -> None:
        self.poss = poss

        self.atoms = None
        self.initiate_atoms()

        self.mass = pd.DataFrame(columns=["mass"])
        self.mass_from_possibility()

        self.order_focal_sets = order_focal_sets
        self.check_order()

    def initiate_atoms(self) -> None:
        """
        Check if the possibility distribution is well-defined
        """
        assert 1 in self.poss.values(), "The possibility distribution must include 1 at least once ! %s" % self.poss
        for k in self.poss.keys():
            assert "," not in k, "The keys cannot contain ',' in it. Please change your key names: %s" % k
            assert "empty" not in k, "The keys cannot contain 'empty' in it. Please change your key names: %s" \
                                     % self.poss.keys()
            assert 0 <= self.poss[k] <= 1, "Possibility distribution should be between 0 and 1: '%s': %s" \
                                           % (k, self.poss[k])
        self.atoms = self.poss.keys()

    def mass_from_possibility(self) -> None:
        """Find the focal sets and compute their mass from a possibility distribution.
        Return a DataFrame with column 'mass'
        Example:
                    mass
        x1,x2,x3    0.2
        x1,x3       0.3
        x1          0.5
        """

        # Create a row for the empty set with null mass for later difference
        self.mass.loc["empty", "mass"] = 0

        # Compute focal sets
        for i, alpha in enumerate(set(self.poss.values())):
            key_focal = [k for k in self.atoms if self.poss[k] >= alpha]
            key_focal = ",".join(key_focal)
            self.mass.loc[key_focal, "mass"] = alpha

        # Compute masses
        self.mass = self.mass.sort_values(by="mass", axis=0, ascending=True)
        self.mass["mass"] = self.mass["mass"].diff()

        self.mass.drop(index="empty", inplace=True)

    def check_order(self) -> None:
        if self.order_focal_sets is not None:
            assert self.mass.index.sort_values().equals(self.order_focal_sets.index.sort_values()), \
                "order_focal_sets.index does not match the one of mass.index." \
                "I am not that smart, please use %s instead of %s"\
                % (self.mass.index.sort_values(), self.order_focal_sets.index.sort_values())

            assert "order" in self.order_focal_sets.columns, "order_focal_sets should have a column named 'order': %s" \
                                                             % self.order_focal_sets.columns


class NecessityBivariate:

    def __init__(self, nec_x: NecessityUnivariate, nec_y: NecessityUnivariate,
                 copula: typing.Callable[[float, float], float]) -> None:
        self.nec_x = nec_x
        self.nec_y = nec_y
        self.copula = copula

        for nec in [self.nec_x, self.nec_y]:
            assert nec.order_focal_sets is not None, "No specified order on focal sets given: %s" % nec.mass

        self.multi_index = pd.MultiIndex.from_product([self.nec_x.mass.index, self.nec_y.mass.index], names=("X", "Y"))
        self.mass = pd.DataFrame(columns=["mass"], index=self.multi_index)
        self.join_mass()

    def join_mass(self) -> None:
        """
        Compute the joint mass from two marginal masses, with a specified order and a specified copula
        mass_x, mass_y: pd.DataFrame with column 'mass'.
        Example
                    mass
        x1,x2,x3    0.2
        x1,x3       0.3
        x1          0.5

        order_x, order_y: pd.DataFrame with column 'order'.
        Example
                        order

        x1              2
        x1,x3           1
        x1,x2,x3        3

        copula: a function that takes as argument two floats between 0 and 1 and returns the copula on those numbers

        pd.DataFrame. A dataframe with MultiIndex (product of Focal sets from mass_x and mass_y)
        and column 'mass'
        """
        for a_x, a_y in self.multi_index:
            index_focal = self.nec_x.order_focal_sets[
                self.nec_x.order_focal_sets["order"] < self.nec_x.order_focal_sets.loc[a_x, "order"]].index
            sum_x_inf = self.nec_x.mass[self.nec_x.mass.index.isin(index_focal.values)]["mass"].sum(axis=0)
            sum_x_sup = sum_x_inf + self.nec_x.mass.loc[a_x, "mass"]

            index_focal = self.nec_y.order_focal_sets[
                self.nec_y.order_focal_sets["order"] < self.nec_y.order_focal_sets.loc[a_y, "order"]].index
            sum_y_inf = self.nec_y.mass[self.nec_y.mass.index.isin(index_focal.values)]["mass"].sum(axis=0)
            sum_y_sup = sum_y_inf + self.nec_y.mass.loc[a_y, "mass"]

            self.mass.loc[(a_x, a_y), "mass"] = \
                self.copula(sum_x_sup, sum_y_sup) - self.copula(sum_x_sup, sum_y_inf) - \
                self.copula(sum_x_inf, sum_y_sup) + self.copula(sum_x_inf, sum_y_inf)
