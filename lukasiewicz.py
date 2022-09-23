import numpy as np

def luka(u,v):
    return np.max([0, u+v-1])

def pos(u):
    return np.max([0, u])

def necessity(a0, a1, b):
    """
    Compute the bivariate necessity function of two possibility measures (see presentation)  as a 3x7 numpy array. 
    It is important to note that the value for y1y2, x1x2x3 is not yet one as this function aims to verify if the necessity functon is well defined.

    :param a0: the first alpha-cut value of X possibility distribution
    :param a1: the second alpha-cut value (>=a0) of X possibility distribution
    :param b: the only alpha cut valut of Y possibility distribution
    :return nec: a (3,7) numpy array with rows corresponding to (y1, y2, y1y2) and columns corrseponding to (x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3)
    """
    nec = np.zeros(shape=(3,7))
    
    nec[1,1] = pos(1-b-a1)
    nec[1,3] = pos(1-b-a1)
    nec[1,5] = pos(1-b-a1) + pos(a1-a0-b)
    nec[1,6] = pos(1-b-a1) + pos(a0-b)
    nec[2,1] = pos(1-b-a1) + pos(b-a1)
    nec[2,3] = pos(1-b-a1) + pos(b-a1)
    nec[2,5] = pos(1-b-a1) + pos(b-a1) + pos(b+a1-a0-1) + pos(a1-a0-b)
    nec[2,6] = pos(1-b-a1) + pos(a1-a0-b) + pos(a0-b) + pos(b-a1) + pos(b+a1-a0-1) + pos(b+a0-1) 
    
    return nec

def sum_mass(a0, a1, b):
    """
    Sum of all mass function terms to verify it equals to 1. 
    """
    m = pos(1-b-a1) + pos(a1-a0-b) + pos(a0-b) + pos(b-a1) + pos(b+a1-a0-1) + pos(b+a0-1)
    return m

if __name__=="__main__":
    min_nec = np.ones(shape=(3,7))
    max_nec = np.zeros(shape=(3,7)) 
    num = 101
    s_min = None
    print(sum_mass(0.5,1,0.5))
    for a0 in np.linspace(0,1,num):
        print("%s / %s" % (int(a0*(num-1)), num-1))
        for a1 in np.linspace(a0,1,num):
            if a1>=a0:
                for b in np.linspace(0,1,num):
                    s = sum_mass(a0, a1, b)
                    if not s_min:
                        s_min = s
                        a0_min, a1_min, b_min = a0, a1, b
                    elif s < s_min:
                        s_min = s
                        a0_min, a1_min, b_min = a0, a1, b
                    #nec = necessity(a0, a1, b)
                    #x = pos(1-b-a1) + pos(a1-a0-b) + pos(a0-b) + pos(b-a1) + pos(b+a1-a0-1) + pos(b+a0-1) 
                    #if x<0.9:
                    #    print("%s a0=%s ; a1 = %s, b=%s"%(x, a0, a1, b))

                    #min_nec = np.minimum(nec, min_nec)
                    #max_nec = np.maximum(nec, max_nec)

    print("Minimum")
    print("%s a0=%s ; a1 = %s, b=%s"%(s_min, a0_min, a1_min, b_min))
