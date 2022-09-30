import itertools
import pandas as pd

from copulas import min_copula
from necessity_functions import NecessityUnivariate, NecessityBivariate
from robust_set_sampling import RobustCredalSetUnivariate, RobustCredalSetBivariate


if __name__ == "__main__":
    # Possibility distributions
    poss_x = {"x1": 0.2, "x2": 1, "x3": 0.2}
    poss_y = {"y1": 0.7, "y2": 1}
    
    order_x_precise = pd.DataFrame(columns=["order"], index=["x1", "x2", "x3"], data=[1, 2, 3])
    order_y_precise = pd.DataFrame(columns=["order"], index=["y1", "y2"], data=[1, 2])
    
    # Finding focal sets
    nec_x_vanilla = NecessityUnivariate(poss_x)
    nec_y_vanilla = NecessityUnivariate(poss_y)

    n_order = 1
    flag_break = False
    # Orderings
    for perm_x in itertools.permutations(range(1, len(nec_x_vanilla.mass.index) + 1)):
        order_x = pd.DataFrame(columns=["order"], index=nec_x_vanilla.mass.index, data=perm_x)
        nec_x = NecessityUnivariate(poss_x, order_x)
        rob_x = RobustCredalSetUnivariate(nec_x)

        for perm_y in itertools.permutations(range(1, len(nec_y_vanilla.mass.index) + 1)):
            order_y = pd.DataFrame(columns=["order"], index=nec_y_vanilla.mass.index, data=perm_y)
            nec_y = NecessityUnivariate(poss_y, order_y)
            rob_y = RobustCredalSetUnivariate(nec_y)

            rob_xy = RobustCredalSetBivariate(rob_x, rob_y, order_x_precise, order_y_precise, min_copula)
            rob_xy.approximate_robust_credal_set()

            nec_xy = NecessityBivariate(nec_x, nec_y, min_copula)

            if ((rob_xy.approximation["P_inf"] - nec_xy.necessity["Nec"]) < - rob_xy.rob_x.epsilon).any():
                nec_xy.necessity.to_csv("%s_Nec_xy.csv" % n_order)

                order_x.append(order_y).to_csv("%s_orders.csv" % n_order)

                rob_xy.approximation.to_csv("%s_P_inf.csv" % n_order)

                n_order += 1
                flag_break = True

            if flag_break:
                break
        if flag_break:
            break
