import numpy as np
import scipy
import prettyprint as pretty
import math


K=50

#definition of kronecker delta and SHO states


def kronDelta(m,n):

    if m == n:
        return 1
    else:
        return 0


def SHO_n(n,x):

    return 1/np.sqrt(np.sqrt(np.pi) * 2**n *math.factorial(n)) * np.exp(-0.5*x**2) * scipy.special.hermite(n, monic=False)(x)


def H_SHO_m_n(m, n, barrier, bias):

    val = 1/4 * ( (2*n+1) * kronDelta(m,n) - np.sqrt(n*(n-1)) * kronDelta(m,n-2) - np.sqrt((n+1)*(n+2)) * kronDelta(m,n+2) ) \
        + 1/(256*barrier) * ( np.sqrt(n*(n-1)*(n-2)*(n-3)) * kronDelta(m,n-4) + 2*(2*n-1)*np.sqrt(n*(n-1))*kronDelta(m,n-2) + 3*(2*n**2+2*n+1)*kronDelta(m,n) + 2*(2*n+3)*np.sqrt((n+1)*(n+2))*kronDelta(m,n+2) + np.sqrt((n+1)*(n+2)*(n+3)*(n+4))*kronDelta(m,n+4) ) \
        - 1/8 * ( (2*n+1) * kronDelta(m,n) + np.sqrt(n*(n-1)) * kronDelta(m,n-2) + np.sqrt((n+1)*(n+2)) * kronDelta(m,n+2) ) \
        - bias/np.sqrt(2) * (np.sqrt(n) * kronDelta(m,n-1) + np.sqrt(n+1) * kronDelta(m,n+1))

    return val


def Q_SHO_m_n(m, n):

    val = np.sqrt(n/2)*kronDelta(m,n-1) + np.sqrt((n+1)/2)*kronDelta(m,n+1)

    return val



def calc_H_DVR_Q_mu(barrier, bias):

    H_SHO_BASIS = np.empty((K,K))

    for i in range(H_SHO_BASIS.shape[0]):
        for j in range(H_SHO_BASIS.shape[1]):
            H_SHO_BASIS[i, j] = H_SHO_m_n(i,j,barrier,bias)

    E_n, H_EIGENVECS = np.linalg.eigh(H_SHO_BASIS)

    H_ENERGY_BASIS = np.diag([E_n[0],E_n[1],E_n[2],E_n[3]])

    Q_SHO_BASIS = np.empty((K,K))

    for i in range(Q_SHO_BASIS.shape[0]):
        for j in range(Q_SHO_BASIS.shape[1]):
            Q_SHO_BASIS[i, j] = Q_SHO_m_n(i,j)

    trafmat_SHO_ENERGY = np.transpose(H_EIGENVECS)

    Q_ENERGY_BASIS_FULL = trafmat_SHO_ENERGY @ Q_SHO_BASIS @ np.linalg.inv(trafmat_SHO_ENERGY)

    Q_ENERGY_BASIS = np.empty((4,4))

    for i in range(Q_ENERGY_BASIS.shape[0]):
        for j in range(Q_ENERGY_BASIS.shape[1]):
            Q_ENERGY_BASIS[i,j] = Q_ENERGY_BASIS_FULL[i,j]

    trafmat_ENERGY_LOC = 1/np.sqrt(2)*np.array([[1,1,0,0],
                                             [-1,1,0,0],
                                             [0,0,1,1],
                                             [0,0,-1,1]])

    H_LOC_BASIS = trafmat_ENERGY_LOC @ H_ENERGY_BASIS @ np.linalg.inv(trafmat_ENERGY_LOC)

    Q_LOC_BASIS = trafmat_ENERGY_LOC @ Q_ENERGY_BASIS @ np.linalg.inv(trafmat_ENERGY_LOC)

    Q_mu, Q_EIGENVECS = np.linalg.eigh(Q_LOC_BASIS)

    trafmat_LOC_DVR = np.transpose(Q_EIGENVECS)

    H_DVR_BASIS = trafmat_LOC_DVR @ H_LOC_BASIS @ np.linalg.inv(trafmat_LOC_DVR)

    Q_DVR_BASIS = trafmat_LOC_DVR @ Q_LOC_BASIS @ np.linalg.inv(trafmat_LOC_DVR)

    return H_DVR_BASIS, Q_mu, E_n, trafmat_SHO_ENERGY, trafmat_ENERGY_LOC, trafmat_LOC_DVR
