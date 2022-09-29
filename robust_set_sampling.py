import pandas as pd
import typing
import itertools
import numpy as np

from necessity_functions import NecessityUnivariate


class RobustCredalSetUnivariate:

    def __init__(self, nec: NecessityUnivariate, epsilon: float = 1e-6, samples_per_interval=11) -> None:
        """
        epsilon: float. Margin of error when verifying that the probability does indeed belong to the credal set
        samples_per_interval: int. The number of point for the np.linspace sampling
        """
        self.nec = nec
        self.epsilon = epsilon
        self.samples_per_interval = samples_per_interval

        self.prob_range = pd.DataFrame(columns=["Nec", "Pl"], index=pd.Index(self.nec.atoms).union(self.nec.mass.index))
        self.compute_prob_range()

        self.generator = self.generator_credal_set()

    def compute_prob_range(self) -> None:
        for event in self.prob_range.index:
            inclusion = [k in event for k in self.nec.mass.index]
            self.prob_range.loc[event, "Nec"] = self.nec.mass.loc[inclusion, "mass"].sum()

            intersection = [len(set(event.split(",")) & set(k.split(","))) > 0 for k in self.nec.mass.index]
            self.prob_range.loc[event, "Pl"] = self.nec.mass[intersection]["mass"].sum()

    def generator_credal_set(self) -> pd.DataFrame:
        """
        Generator function for probabilities. Yields a sampled probability respecting the probability ranges on atoms
        """

        # TODO How to do the same with any number of atoms?
        p = pd.DataFrame(columns=["P"], index=pd.Index(self.nec.atoms), dtype=float)
        for x1 in np.linspace(self.prob_range.loc[p.index[0], "Nec"], self.prob_range.loc[p.index[0], "Pl"],
                              num=self.samples_per_interval):
            for x2 in np.linspace(self.prob_range.loc[p.index[1], "Nec"], self.prob_range.loc[p.index[1], "Pl"],
                                  num=self.samples_per_interval):
                p.loc[p.index[0:3], "P"] = [x1, x2, 1 - x1 - x2]

                if np.any(p["P"] < 0):
                    continue

                for focal_set in self.nec.mass.index:
                    # Because it is from a Necessity function, checking on focal sets is enough
                    if p.loc[focal_set.split(","), "P"].sum() < self.prob_range.loc[focal_set, "Nec"] - self.epsilon:
                        continue
                yield p


class RobustCredalSetBivariate:

    def __init__(self, rob_x: RobustCredalSetUnivariate, rob_y: RobustCredalSetUnivariate, order_x_p: pd.DataFrame,
                 order_y_p: pd.DataFrame, copula: typing.Callable[[float, float], float]) -> None:
        """
        order_x_p, order_y_p: pd.DataFrame. order on atoms ('precise' order). Rows are atoms, column is 'order'
        """

        self.rob_x = rob_x
        self.rob_y = rob_y
        self.copula = copula

        self.order_x_p = order_x_p
        self.order_y_p = order_y_p

        self.p_xy = pd.DataFrame(columns=["P"],
                                 index=pd.MultiIndex.from_product([self.rob_x.nec.atoms, self.rob_y.nec.atoms],
                                                                  names=["X", "Y"]))

        self.approximation = None

    def join_proba_on_atoms(self, p_x: pd.DataFrame, p_y: pd.DataFrame) -> None:
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
        """
        for a_x, a_y in self.p_xy.index:
            index = self.order_x_p[self.order_x_p["order"] < self.order_x_p.loc[a_x, "order"]].index
            sum_x_inf = p_x.loc[index, "P"].sum(axis=0)
            sum_x_sup = sum_x_inf + float(p_x.loc[a_x, "P"])

            index = self.order_y_p[self.order_y_p["order"] < self.order_y_p.loc[a_y, "order"]].index
            sum_y_inf = p_y.loc[index, "P"].sum(axis=0)
            sum_y_sup = sum_y_inf + float(p_y.loc[a_y, "P"])

            self.p_xy.loc[(a_x, a_y), "P"] = self.copula(sum_x_sup, sum_y_sup) - self.copula(sum_x_sup, sum_y_inf) - \
                                             self.copula(sum_x_inf, sum_y_sup) + self.copula(sum_x_inf, sum_y_inf)

    def approximate_robust_credal_set(self) -> None:
        """
        Approximate the robust credal set from marginal credal sets
        Output robust_df: A DataFrame with multi index and a single column "P_inf" containing the approximation of the lower
        probability on events.

        We generate sampled marginal probabilities from 'prob_range' and compute the joint_proba_on_atoms from them.
        Then we compare that to robust_df and keep the lowest proba for every event.
        """

        full_events_x = []
        # self.order_x_p.keys() is basically ["x1", "x2", "x3"] (=X). full_events_x is thus the power set of X
        for k in range(1, len(self.order_x_p.keys()) + 1):
            full_events_x += [",".join(list(j)) for j in itertools.combinations(self.order_x_p.keys(), k)]

        full_events_y = []
        # self.order_y_p.keys() is basically ["y1", "y2", "y3"] (=Y). full_events_y is thus the power set of Y
        for k in range(1, len(self.order_y_p.keys()) + 1):
            full_events_y += [",".join(list(j)) for j in itertools.combinations(self.order_x_p.keys(), k)]

        multi = pd.MultiIndex.from_product([full_events_x, full_events_y], names=["X", "Y"])
        self.approximation = pd.DataFrame(columns=["P_inf", "P"], index=multi)
        self.approximation["P_inf"] = 1

        generator_x = self.rob_x.generator_credal_set()
        generator_y = self.rob_y.generator_credal_set()

        for p_x in generator_x:
            for p_y in generator_y:
                self.join_proba_on_atoms(p_x, p_y)
                for x, y in self.approximation.index:
                    x_i, y_i = x.split(","), y.split(",")
                    atoms = list(itertools.product(*[x_i, y_i]))
                    if self.approximation.loc[(x, y), "P_inf"] > self.p_xy.loc[atoms, "P"].sum():
                        self.approximation.loc[(x, y), "P_inf"] = np.round(self.p_xy.loc[atoms, "P"].sum(), 6)
                        self.approximation.loc[(x, y), "P"] = str(list(p_x["P"].round(6))) + " | " + str(list(p_y["P"].round(6)))
