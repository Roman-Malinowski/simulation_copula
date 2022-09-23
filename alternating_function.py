import numpy as np
import itertools


def C(u, v):
    """
    Enter here the formula for your copula
    :param u: The first parameter of the copula, should be between 0 and 1
    :param v: The second parameter of the copula, should be between 0 and 1
    :return c: The value of the copula at coordinates u,v
    """
    assert 0 <= u <= 1
    assert 0 <= v <= 1

    # c = np.max([0, u+v-1])
    c = np.min([u, v])

    return c


def pos(u):
    return np.max([0, u])


def compute_P_sup(alpha, beta):
    """
    Upper joint probability for proba with two atoms for X and two atoms for Y
    pi(x2) = alpha
    pi(y1) = beta
    """
    # # CAREFUL This is for the min(u,v)  Copula
    # Px1y1 = beta
    # Px2y1 = beta - np.min([beta, 1-alpha])
    # Px1y2 = 1
    # Px2y2 = alpha

    # CAREFUL This is for the max(u+v-1,0) Copula
    Px1y1 = 1
    Px2y1 = np.min([alpha, beta])
    Px1y2 = 1
    Px2y2 = alpha
    P_sup_ = np.array([[Px1y1, Px1y2], [Px2y1, Px2y2]])

    return P_sup_


def compute_P_sup(alpha, beta1, beta2):
    """
    Upper joint probability for proba with two atoms for X and three atoms for Y
    pi(x2) = alpha
    pi(y1) = beta1 < beta2 = pi(y3)
    """
    # CAREFUL This is for the min(u,v)  Copula
    # Px1y1 = beta1
    # Px2y1 = beta1 - np.min([1 - alpha, beta1])
    # Px1y2 = 1
    # Px2y2 = 1
    # Px1y3 = beta2
    # Px2y3 = 1 - np.max([1 - beta2, 1 - alpha])

    # CAREFUL This is for the max(u+v-1,0) Copula
    Px1y1 = beta1
    Px2y1 = np.min([alpha, beta1])
    Px1y2 = 1
    Px2y2 = alpha
    Px1y3 = beta2
    Px2y3 = 0

    P_sup_ = np.array([[Px1y1, Px1y2, Px1y3], [Px2y1, Px2y2, Px2y3]])

    return P_sup_


def alternating(P_sup_, A, B, threshold=0):
    """
    Test if the upper probability is alternating or not
    :param P_sup: a (2,2) array with the value of the upper probability for the atoms
    :param A: a list of (i,j) coordinates, without duplicates, representing the first event
    :param B: a list of (i,j) coordinates, without duplicates, representing the second event
    :return P_alt: a boolean. True if P is 2 alternating, False otherwise
    :return P_AUB, P_AintB, P_A, P_B: floats. The value of the probabilities considered in the inequality
    """
    AUB = set(A).union(set(B))
    AintB = set(A).intersection(set(B))

    P_AUB = 0
    for ij in AUB:
        P_AUB += P_sup_[int(ij[0]), int(ij[1])]

    P_AintB = 0
    for ij in AintB:
        P_AintB += P_sup_[int(ij[0]), int(ij[1])]

    P_A = 0
    for ij in set(A):
        P_A += P_sup_[int(ij[0]), int(ij[1])]

    P_B = 0
    for ij in set(B):
        P_B += P_sup_[int(ij[0]), int(ij[1])]

    return P_AUB + P_AintB <= P_A + P_B + threshold, P_AUB, P_AintB, P_A, P_B


def check_all_events(prec, dim_x=2, dim_y=2, copula="", threshold=0.0):
    if dim_x == 2 and dim_y == 2:
        # create a list of "ij" for all atoms to access P_sup[i,j]
        coordinates = [str(i) + str(j) for i in range(dim_x) for j in range(dim_y)]

        for alpha in np.linspace(0, 1, prec + 1):
            for beta in np.linspace(0, 1, prec + 1):
                # Creating a random joint upper probability from the Copula applied to the CDFs for possibilities
                P_sup = compute_P_sup(alpha, beta)
                # TODO Optimize the choice of permutations so no duplicates are considered
                # Enumerating all events A and B composed of atoms "ij"
                for k in range(dim_x ** dim_y):
                    A_iter = list(itertools.combinations(coordinates, k + 1))
                    # For instance if k=2, A will be the list of all events composed of 2 atoms represented by a string "ij"
                    # We only need to check events B with a lower number of elements than A
                    for j in range(k + 1):
                        B_iter = list(itertools.combinations(coordinates, j + 1))
                        for A in A_iter:
                            for B in B_iter:
                                is_alternating, P_AUB, P_AintB, P_A, P_B = alternating(P_sup, A, B, threshold=threshold)
                                if not is_alternating:
                                    with open("results.txt", "a") as f:
                                        f.write(
                                            "C: %s | alpha: %s | beta: %s | A: %s | B: %s | P_AUB + P_AintB = %s > %s = P_A + P_B\n" % (
                                                copula, alpha, beta, A, B, P_AUB + P_AintB, P_A + P_B))
                                    print("Written line!")

    elif dim_x == 2 and dim_y == 3:
        # create a list of "ij" for all atoms to access P_sup[i,j]
        coordinates = [str(i) + str(j) for i in range(dim_x) for j in range(dim_y)]

        for alpha in np.linspace(0, 1, prec + 1):
            for beta1 in np.linspace(0, 1, prec + 1):
                for beta2 in np.linspace(beta1, 1, int((1 - beta1) * prec) + 1):
                    # Creating a random joint upper probability from the Copula applied to the CDFs for possibilities
                    P_sup = compute_P_sup(alpha, beta1, beta2)
                    # TODO Optimize the choice of permutations so no duplicates are considered
                    # Enumerating all events A and B composed of atoms "ij"
                    for k in range(dim_x ** dim_y):
                        A_iter = list(itertools.combinations(coordinates, k + 1))
                        # For instance if k=2, A will be the list of all events composed of 2 atoms represented by a string "ij"
                        # We only need to check events B with a lower number of elements than A
                        for j in range(k + 1):
                            B_iter = list(itertools.combinations(coordinates, j + 1))
                            for A in A_iter:
                                for B in B_iter:
                                    is_alternating, P_AUB, P_AintB, P_A, P_B = alternating(P_sup, A, B,
                                                                                           threshold=threshold)
                                    if not is_alternating:
                                        with open("results.txt", "a") as f:
                                            f.write(
                                                "C: %s | alpha: %s | beta1: %s | beta2: %s | A: %s | B: %s | P_AUB + P_AintB = %s > %s = P_A + P_B\n" % (
                                                    copula, alpha, beta1, beta2, A, B, P_AUB + P_AintB, P_A + P_B))
                                        print("Written line!")


if __name__ == "__main__":
    with open("results.txt", "w") as f:
        f.write("")

    precision = 100

    copula_used = "max(u+v-1,0)"

    check_all_events(precision, dim_x=2, dim_y=3, copula=copula_used, threshold=0.00001)
