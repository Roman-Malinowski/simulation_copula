import itertools
import pandas as pd
import os.path
import numpy as np

from copulas import min_copula
from necessity_functions import NecessityUnivariate, NecessityBivariate
from robust_set_sampling import RobustCredalSetUnivariate, RobustCredalSetBivariate, IndexSampling


def generator_poss(keys: list) -> dict:
    list_of_ranges = np.array([np.linspace(0, 1, 11) for _ in range(len(keys) - 1)])
    k_index = IndexSampling([len(k) for k in list_of_ranges])
    for k in range(len(keys)):
        for i in range(np.power(list_of_ranges.shape[1], list_of_ranges.shape[0])):
            a = np.insert([list_of_ranges[j, k_index.index[j]] for j in range(len(list_of_ranges))], k, 1)
            k_index.next()
            yield {keys[ind]: np.round(a[ind], 3) for ind in range(len(keys))}


if __name__ == "__main__":
    output_dir = "/work/scratch/malinoro/simulation_copula/out"
    # Possibility distributions
    possibilities_x = generator_poss(["x1", "x2", "x3", "x4"])
    possibilities_y = generator_poss(["y1", "y2", "y3", "y4"])

    order_x_precise = pd.DataFrame(columns=["order"], index=["x1", "x2", "x3", "x4"], data=[1, 2, 3, 4])
    order_y_precise = pd.DataFrame(columns=["order"], index=["y1", "y2", "y3", "y4"], data=[1, 2, 3, 4])

    for poss_x in possibilities_x:
        for poss_y in possibilities_y:
            # Finding focal sets
            nec_x_vanilla = NecessityUnivariate(poss_x)
            nec_y_vanilla = NecessityUnivariate(poss_y)

            n_order = 1
            # Orderings
            flag_order_work = False

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

                    if ((rob_xy.approximation["P_inf"] - nec_xy.necessity["Nec"]) > - rob_xy.rob_x.epsilon).all():
                        flag_order_work = True
                        continue
                if flag_order_work:
                    continue

            if not flag_order_work:
                nec_xy.nec_x.mass.to_csv(os.path.join(output_dir, "%s_Nec_x.csv" % n_order))
                nec_xy.nec_y.mass.to_csv(os.path.join(output_dir, "%s_Nec_y.csv" % n_order))

                rob_xy.approximation.to_csv(os.path.join(output_dir, "%s_P_inf.csv" % n_order))

                n_order += 1

        possibilities_y = generator_poss(["y1", "y2", "y3"])
