import numpy as np
import scipy
import math
import helpful_functions as hf


K=50


def calc_H_DVR_Q_DVR(n_sys, barrier, bias):

    H_SHO_BASIS = np.empty((K,K))

    for m in range(H_SHO_BASIS.shape[0]):
        for n in range(H_SHO_BASIS.shape[1]):
            H_SHO_BASIS[m, n] = 1/4 * ( (2*n+1) * hf.kronecker_delta(m,n) - np.sqrt(n*(n-1)) * hf.kronecker_delta(m,n-2) - np.sqrt((n+1)*(n+2)) * hf.kronecker_delta(m,n+2) ) \
                                + 1/(256*barrier) * ( np.sqrt(n*(n-1)*(n-2)*(n-3)) * hf.kronecker_delta(m,n-4) + 2*(2*n-1)*np.sqrt(n*(n-1))*hf.kronecker_delta(m,n-2) + 3*(2*n**2+2*n+1)*hf.kronecker_delta(m,n) \
                                + 2*(2*n+3)*np.sqrt((n+1)*(n+2))*hf.kronecker_delta(m,n+2) + np.sqrt((n+1)*(n+2)*(n+3)*(n+4))*hf.kronecker_delta(m,n+4) ) \
                                - 1/8 * ( (2*n+1) * hf.kronecker_delta(m,n) + np.sqrt(n*(n-1)) * hf.kronecker_delta(m,n-2) + np.sqrt((n+1)*(n+2)) * hf.kronecker_delta(m,n+2) ) \
                                - bias/np.sqrt(2) * (np.sqrt(n) * hf.kronecker_delta(m,n-1) + np.sqrt(n+1) * hf.kronecker_delta(m,n+1))

    E_n, H_EIGENVECS = np.linalg.eigh(H_SHO_BASIS)

    H_ENERGY_BASIS = np.diag([E_n[0],E_n[1],E_n[2],E_n[3]])

    Q_SHO_BASIS = np.empty((K,K))

    for m in range(Q_SHO_BASIS.shape[0]):
        for n in range(Q_SHO_BASIS.shape[1]):
            Q_SHO_BASIS[m, n] = np.sqrt(n/2)*hf.kronecker_delta(m,n-1) + np.sqrt((n+1)/2)*hf.kronecker_delta(m,n+1)

    trafmat_SHO_ENERGY = np.transpose(H_EIGENVECS)

    Q_ENERGY_BASIS_FULL = trafmat_SHO_ENERGY @ Q_SHO_BASIS @ np.linalg.inv(trafmat_SHO_ENERGY)

    Q_ENERGY_BASIS = np.empty((4,4))

    for i in range(Q_ENERGY_BASIS.shape[0]):
        for j in range(Q_ENERGY_BASIS.shape[1]):
            Q_ENERGY_BASIS[i,j] = Q_ENERGY_BASIS_FULL[i,j]

    trafmat_ENERGY_LOC = 1/np.sqrt(2)*np.array([[1,-1,0,0],
                                             [1,1,0,0],
                                             [0,0,1,-1],
                                             [0,0,1,1]])

    H_LOC_BASIS = trafmat_ENERGY_LOC @ H_ENERGY_BASIS @ np.linalg.inv(trafmat_ENERGY_LOC)

    Q_LOC_BASIS = trafmat_ENERGY_LOC @ Q_ENERGY_BASIS @ np.linalg.inv(trafmat_ENERGY_LOC)

    Q_mu, Q_EIGENVECS = np.linalg.eigh(Q_LOC_BASIS)

    trafmat_LOC_DVR = np.transpose(Q_EIGENVECS)

    H_DVR_BASIS_pre = trafmat_LOC_DVR @ H_LOC_BASIS @ np.linalg.inv(trafmat_LOC_DVR)

    Q_DVR_BASIS_pre = trafmat_LOC_DVR @ Q_LOC_BASIS @ np.linalg.inv(trafmat_LOC_DVR)

    H_DVR_BASIS = hf.kronecker_product_with_identity(n_sys,H_DVR_BASIS_pre)

    Q_DVR_BASIS = hf.kronecker_product_with_identity(n_sys,Q_DVR_BASIS_pre)

    return H_DVR_BASIS, Q_DVR_BASIS, E_n, trafmat_SHO_ENERGY, trafmat_ENERGY_LOC, trafmat_LOC_DVR
