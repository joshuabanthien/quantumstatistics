import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import rate_matrix


#solves diff.eq. via 1st-order runge kutta

def runge_kutta_1st_order(f, T, Y0, M):

    n = len(T)

    Y = np.zeros((n, len(Y0)))

    Y[0] = Y0

    for i in range(n - 1):

        Y[i+1] = Y[i] + (T[i+1] - T[i]) * f(Y[i], T[i], M)

    return Y
