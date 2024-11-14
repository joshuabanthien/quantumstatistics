import numpy as np
import scipy
import math_collection

K=50


#calculate Hamiltonian and position operator in DVR-basis and further transformation matrices etc

def H_DVR_Q_DVR(n_sys, barrier, bias):

    H_SHO_BASIS = np.empty((K,K))

    for m in range(H_SHO_BASIS.shape[0]):

        for n in range(H_SHO_BASIS.shape[1]):

            H_SHO_BASIS[m, n] = 1/4 * ( (2*n+1) * math_collection.kronecker_delta(m,n) - np.sqrt(n*(n-1)) * math_collection.kronecker_delta(m,n-2) - np.sqrt((n+1)*(n+2)) * math_collection.kronecker_delta(m,n+2) ) \
                                + 1/(256*barrier) * ( np.sqrt(n*(n-1)*(n-2)*(n-3)) * math_collection.kronecker_delta(m,n-4) + 2*(2*n-1)*np.sqrt(n*(n-1))*math_collection.kronecker_delta(m,n-2) + 3*(2*n**2+2*n+1)*math_collection.kronecker_delta(m,n) \
                                + 2*(2*n+3)*np.sqrt((n+1)*(n+2))*math_collection.kronecker_delta(m,n+2) + np.sqrt((n+1)*(n+2)*(n+3)*(n+4))*math_collection.kronecker_delta(m,n+4) ) \
                                - 1/8 * ( (2*n+1) * math_collection.kronecker_delta(m,n) + np.sqrt(n*(n-1)) * math_collection.kronecker_delta(m,n-2) + np.sqrt((n+1)*(n+2)) * math_collection.kronecker_delta(m,n+2) ) \
                                - bias/np.sqrt(2) * (np.sqrt(n) * math_collection.kronecker_delta(m,n-1) + np.sqrt(n+1) * math_collection.kronecker_delta(m,n+1))

    E_N, H_EIGENVECS = np.linalg.eigh(H_SHO_BASIS)

    H_ENERGY_BASIS = np.diag([E_N[0],E_N[1],E_N[2],E_N[3]])

    Q_SHO_BASIS = np.empty((K,K))

    for m in range(Q_SHO_BASIS.shape[0]):

        for n in range(Q_SHO_BASIS.shape[1]):

            Q_SHO_BASIS[m, n] = np.sqrt(n/2)*math_collection.kronecker_delta(m,n-1) + np.sqrt((n+1)/2)*math_collection.kronecker_delta(m,n+1)

    TRAFMAT_SHO_ENERGY = np.transpose(H_EIGENVECS)

    Q_ENERGY_BASIS_FULL = TRAFMAT_SHO_ENERGY @ Q_SHO_BASIS @ np.linalg.inv(TRAFMAT_SHO_ENERGY)

    Q_ENERGY_BASIS = np.empty((4,4))

    for i in range(Q_ENERGY_BASIS.shape[0]):

        for j in range(Q_ENERGY_BASIS.shape[1]):

            Q_ENERGY_BASIS[i,j] = Q_ENERGY_BASIS_FULL[i,j]

    TRAFMAT_ENERGY_LOC = 1/np.sqrt(2)*np.array([[1,-1,0,0],
                                             [1,1,0,0],
                                             [0,0,1,-1],
                                             [0,0,1,1]])

    H_LOC_BASIS = TRAFMAT_ENERGY_LOC @ H_ENERGY_BASIS @ np.linalg.inv(TRAFMAT_ENERGY_LOC)

    Q_LOC_BASIS = TRAFMAT_ENERGY_LOC @ Q_ENERGY_BASIS @ np.linalg.inv(TRAFMAT_ENERGY_LOC)

    Q_MU, Q_EIGENVECS = np.linalg.eigh(Q_LOC_BASIS)

    TRAFMAT_LOC_DVR = np.transpose(Q_EIGENVECS)

    H_DVR_BASIS_PRE = TRAFMAT_LOC_DVR @ H_LOC_BASIS @ np.linalg.inv(TRAFMAT_LOC_DVR)

    Q_DVR_BASIS_PRE = TRAFMAT_LOC_DVR @ Q_LOC_BASIS @ np.linalg.inv(TRAFMAT_LOC_DVR)

    H_DVR_BASIS = math_collection.kronecker_product_with_identity(n_sys,H_DVR_BASIS_PRE)

    Q_DVR_BASIS = math_collection.kronecker_product_with_identity(n_sys,Q_DVR_BASIS_PRE)

    return H_DVR_BASIS, Q_DVR_BASIS, E_N, TRAFMAT_SHO_ENERGY, TRAFMAT_ENERGY_LOC, TRAFMAT_LOC_DVR
