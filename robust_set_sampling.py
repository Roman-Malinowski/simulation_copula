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
        
        self.samples = self.generate_samples()
        self.generator = self.generator_credal_set()

    def compute_prob_range(self) -> None:
        for event in self.prob_range.index:
            self.prob_range.loc[event, "Nec"] = self.nec.necessity.loc[event, "Nec"]

            intersection = [len(set(event.split(",")) & set(k.split(","))) > 0 for k in self.nec.mass.index]
            self.prob_range.loc[event, "Pl"] = self.nec.mass[intersection]["mass"].sum()
    
    
    def generate_samples(self) -> pd.DataFrame:
        """
        Functions that generate Probabilities sampled from the credal set.
        It discretizes the probability range on atoms (linspace(Nec, Pl, samples_per_interval)
        It then do a cartesian product on those discretized ranges, and filter incoherent probabilities
        It also filters probabilities that does not respect Nec condition on the focal sets (because it comes from a possibility, it is sufficient)
        Return a pd.DataFrame with columns being the atoms and each row a different probability mass.
            x1   x2   x3
        0  0.5  0.2  0.3
        1  0.2  0.2  0.6
        2  0.1  0.7  0.2

        """
                # We don't care about Nec and Pl of the final atom because it will be set to 1-P(x1,...,xn-1)
        list_of_ranges = [np.linspace(self.prob_range.loc[atom, "Nec"], self.prob_range.loc[atom, "Pl"],
                                      num=self.samples_per_interval) for k in list(self.nec.atoms)[:-1]]
        
        cartesian_product = np.empty([len(a) for a in list_of_ranges] + [len(list_of_ranges)], dtype=float)  # Creates an empty array with correct dimensions
        for i, a in enumerate(np.ix_(*list_of_ranges)):
            cartesian_product[..., i] = a
        cartesian_product = cartesian_product.reshape(-1, len(list_of_ranges))  # Reshaping so it is of shape (samples_per_interval**(n-1), n-1)
        
        cartesian_product = np.hstack((cartesian_product, 1-np.expand_dims(np.sum(cartesian_product, axis=1), axis=1)))  # Adding the value of the last atom because P is normalized
        cartesian_product = cartesian_product[np.all(0 <= cartesian_product, axis=1) & np.all(cartesian_product<=1, axis=1)]  # Keeping only rows where P is in [0,1]
        
        p = pd.DataFrame(columns=pd.Index(self.nec.atoms), data=cartesian_product, dtype=float)
        
        # Removing P that are not respecting the necessity condition on focal sets 
        for focal_set in self.nec.mass.index:
            nec = self.prob_range.loc[focal_set, "Nec"] - self.epsilon
            p = p[p.loc[:, focal_set.split(",")].sum(axis=1) >= nec]
        return p
    
    
    def generator_credal_set(self) -> pd.DataFrame:
        """
        Generator function for probabilities. Yields a sampled probability respecting the probability ranges on atoms
        """
        for k in range(self.samples.index):
            yield p.loc[k, :]


class RobustCredalSetBivariate:
    def __init__(self, rob_x: RobustCredalSetUnivariate, rob_y: RobustCredalSetUnivariate, order_x_p: pd.DataFrame,
                 order_y_p: pd.DataFrame, copula: typing.Callable[[float, float], float]) -> None:
        """
        order_x_p, order_y_p: pd.DataFrame. order on atoms ('precise' order). Rows are atoms, column is 'order'
        """

        self.rob_x = rob_x
        self.rob_y = rob_y
        self.copula = np.vectorize(copula)

        self.order_x_p = order_x_p
        self.order_y_p = order_y_p
        
        self.p_xy = None
        self.join_proba_on_atoms()

        self.approximation = None
        
        
    def join_proba_on_atoms(self) -> None:
        """
        Compute the joint probabilities on atoms from two marginal probabilities,
        with a specified order and a specified copula
        order_x_p, order_y_p: pd.DataFrame with column 'order'.
        Example
                order
        x3        3
        x1        1
        x2        2
        """
        
        self.order_x_p = self.order_x_p.sort_values(["order"])
        self.order_y_p = self.order_y_p.sort_values(["order"])
        
        # Cumulated distribution functions
        c_px = self.rob_x.samples.copy()
        c_py = self.rob_y.samples.copy()
        
        # TODO CARTESIAN PRODUCT OF THOSE TWO 
        c_px["empty"] = 0
        c_py["empty"] = 0
        
        c_px = c_px[["empty"] + self.order_x_p.index.to_list()].cumsum(axis=1)
        c_py = c_py[["empty"] + self.order_y_p.index.to_list()].cumsum(axis=1)
        
        self.p_xy = pd.DataFrame(columns=pd.MultiIndex.from_product([self.order_x_p.index, self.order_y_p.index], names=["X", "Y"]))
        
        for a_x, a_y in self.p_xy.columns:  # atoms
            i, j = c_px.columns.get_loc(a_x), c_py.columns.get_loc(a_y)
            self.p_xy[(a_x, a_y)] = self.copula(c_px.columns[i].to_numpy(), c_py.columns[j].to_numpy()) + self.copula(c_px.columns[i-1].to_numpy(), c_py.columns[j-1].to_numpy()) - self.copula(c_px.columns[i-1].to_numpy(), c_py.columns[j].to_numpy()) - self.copula(c_px.columns[i].to_numpy(), c_py.columns[j-1].to_numpy())
            

            
    def approximate_robust_credal_set(self) -> None:
        """
        Approximate the robust credal set from the joint samples
        Output robust_df: A DataFrame with multi index and a single column "P_inf" containing the approximation of the lower
        probability on events.
        """

        full_events_x = []
        # self.order_x_p.index is basically ["x1", "x2", "x3"] (=X). full_events_x is thus the power set of X
        for k in range(1, len(self.order_x_p.index) + 1):
            full_events_x += [",".join(list(j)) for j in itertools.combinations(self.order_x_p.index, k)]

        full_events_y = []
        # self.order_y_p.index is basically ["y1", "y2", "y3"] (=Y). full_events_y is thus the power set of Y
        for k in range(1, len(self.order_y_p.index) + 1):
            full_events_y += [",".join(list(j)) for j in itertools.combinations(self.order_y_p.index, k)]

        multi = pd.MultiIndex.from_product([full_events_x, full_events_y], names=["X", "Y"])
        self.approximation = pd.DataFrame(columns=["P_inf", "P"], index=multi)

        for x, y in self.approximation.index:
            x_i, y_i = x.split(","), y.split(",")
            atoms = list(itertools.product(*[x_i, y_i]))
            
            p_inf = self.p_xy.loc[:, atoms].sum(axis=1)
            ind = p_inf.argmin()
            self.approximation.loc[(x, y), "P_inf"] = np.round(p_inf.min(), 6)
            self.rob_x.samples.loc[ind, :
            self.approximation.loc[(x, y), "P"] = str(list(p_x["P"].round(6))) + " | " + str(list(p_y["P"].round(6)))