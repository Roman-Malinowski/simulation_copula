import itertools
import pandas as pd
import os.path
import numpy as np
from numpy.random import default_rng
import logging

from copulas import min_copula, lukaciewicz_copula
from necessity_functions import NecessityUnivariate, NecessityBivariate
from robust_set_sampling import RobustCredalSetUnivariate, RobustCredalSetBivariate, IndexSampling

def random_generator_poss(keys: list, seed=1) -> dict:
    rng = default_rng(seed)
    n = len(keys)
    values = np.zeros(n)
    while True:
        n_ones = rng.choice(n, size=rng.choice(range(1, n), p=[0.7] + [0.3/(n-2)]*(n-2)), replace=False)
        rand = [np.round(rng.random(), 2) for _ in range(n-len(n_ones))]
        values[n_ones] = 1.0
        mask = np.ones(values.shape, dtype=bool)
        mask[n_ones] = False
        values[mask] = rand
        yield {keys[ind]: values[ind] for ind in range(n)}


if __name__ == "__main__":
    output_dir = "/work/scratch/malinoro/simulation_copula/out"
    output_file = "random_orders_N4.csv"
    
    logging.basicConfig(filename=os.path.join(output_dir, output_file.split(".csv")[0] + ".log"), format="%(asctime)s | %(levelname)s: %(message)s", level=logging.DEBUG)

    logging.info("Starting the log file") 
    
    x_space = ["x1", "x2", "x3", "x4"]
    y_space = ["y1", "y2", "y3", "y4"]
    # x_space = ["x1", "x2", "x3"]
    # y_space = ["y1", "y2", "y3"]
    
    copula = lukaciewicz_copula

    # Possibility distributions
    possibilities_x = random_generator_poss(x_space)
    possibilities_y = random_generator_poss(y_space, seed=2)

    order_x_precise = pd.DataFrame(columns=["order"], index=x_space, data=range(1, len(x_space) + 1))
    order_y_precise = pd.DataFrame(columns=["order"], index=y_space, data=range(1, len(y_space) + 1))
    logging.info("Order X: " + str(order_x_precise))
    logging.info("Order Y: " + str(order_y_precise))

    # Initializing the output file
    multi_col = pd.MultiIndex.from_tuples(list(
        zip(["poss"] * (len(x_space) + len(y_space)) + ["focal_sets", "focal_sets"],
            x_space + y_space + ["X", "Y"])), names=["Object", "Space"])
    multi_index = pd.MultiIndex.from_product([pd.Index(name = "poss", data=[]), pd.Index(name="order", data=[])])
    
    final_df = pd.DataFrame(columns=multi_col, index=multi_index)
    final_df.to_csv(os.path.join(output_dir, output_file))

    n_poss = -1
    
    for poss_x, poss_y in zip(possibilities_x, possibilities_y):
        logging.info("Poss X: ", str(poss_x)) 
        flag_skip = np.all([k==0. or k==1. for k in poss_x.values()])
        if flag_skip:
            logging.info("Skipping this possibility")
            continue
        
        logging.info("Poss Y: ", str(poss_y)) 
        flag_skip = np.all([k==0. or k==1. for k in poss_y.values()])
        if flag_skip:
            logging.info("Skipping this possibility")
            flag_skip = np.all([k==0. or k==1. for k in poss_x.values()])
            continue
             
        # Now that we have a valid poss_x and poss_y
        n_poss += 1
        
        # Finding focal sets
        nec_x_vanilla = NecessityUnivariate(poss_x)
        rob_x = RobustCredalSetUnivariate(nec_x_vanilla, samples_per_interval=5)

        nec_y_vanilla = NecessityUnivariate(poss_y)
        rob_y = RobustCredalSetUnivariate(nec_y_vanilla, samples_per_interval=5)

        rob_xy = RobustCredalSetBivariate(rob_x, rob_y, order_x_precise, order_y_precise, copula)
        logging.info("Computing robust credal set...")
        rob_xy.approximate_robust_credal_set()

        flag_order_work = False

        n_order = 0
        logging.info("Starting permutations") 
        for perm_x in itertools.permutations(range(1, len(nec_x_vanilla.mass.index) + 1)):
            order_x = pd.DataFrame(columns=["order"], index=nec_x_vanilla.mass.index, data=perm_x)
            nec_x = NecessityUnivariate(poss_x, order_x)

            for perm_y in itertools.permutations(range(1, len(nec_y_vanilla.mass.index) + 1)):
                order_y = pd.DataFrame(columns=["order"], index=nec_y_vanilla.mass.index, data=perm_y)
                nec_y = NecessityUnivariate(poss_y, order_y)

                nec_xy = NecessityBivariate(nec_x, nec_y, copula)

                if ((rob_xy.approximation["P_inf"] - nec_xy.necessity["Nec"]) > - rob_xy.rob_x.epsilon).all():
                    flag_order_work = True
                    final_df.loc[(n_poss, n_order), [("poss", k) for k in poss_x.keys()]] = poss_x.values()
                    final_df.loc[(n_poss, n_order), [("poss", k) for k in poss_y.keys()]] = poss_y.values()
                    final_df.loc[(n_poss, n_order), [("focal_sets", "X")]] = "<".join(
                        order_x.sort_values(axis=0, by=["order"]).index)
                    final_df.loc[(n_poss, n_order), [("focal_sets", "Y")]] = "<".join(
                        order_y.sort_values(axis=0, by=["order"]).index)
                    logging.info("Writing premutation %s" % n_order)
                    final_df.to_csv(os.path.join(output_dir, output_file), mode='a', header=False)
                    final_df.drop(axis=0, labels=[(n_poss, n_order)], inplace=True)
                    n_order += 1

        if not flag_order_work:
            logging.debug("No order is working!")
            nec_xy.nec_x.mass.to_csv(os.path.join(output_dir, "%s_%s_Nec_x.csv" % (n_poss, n_order)))
            nec_xy.nec_y.mass.to_csv(os.path.join(output_dir, "%s_%s_Nec_y.csv" % (n_poss, n_order)))

            rob_xy.approximation.to_csv(os.path.join(output_dir, "%s_P_inf.csv" % n_poss))

            n_order += 1
