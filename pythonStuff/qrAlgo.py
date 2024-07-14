import numpy as np
import scipy as sci
import math
import cmath

testMat = np.array([[ 0.07660059,  0.06279331, -0.00183684,  0.00229439],
       [-0.07385856, -0.06332512, -0.00697304, -0.00207069],
       [ 0.00036747, -0.00410351, -0.09686056, -0.09182672],
       [-0.00310951,  0.00463532,  0.10567044,  0.09160303]])

def QR_Decomp(A):

    m,n = A.shape

    Q = np.empty((n,n), dtype=complex)
    u = np.empty((n,n), dtype=complex)

    u[:,0] = A[:,0]
    Q[:,0] = u[:,0]/np.linalg.norm(u[:,0])

    for i in range(1,n):

        u[:,i] = A[:,i]
        for j in range(i):
            u[:, i] -= (A[:,i] @ Q[:,j])*Q[:,j]

        Q[:,i] = u[:,i]/np.linalg.norm(u[:,i])

    R = np.zeros((n,m), dtype=complex)

    for i in range(n):
        for j in range(i,m):
            R[i,j] = A[:,j] @ Q[:,i]

    return Q, R


def QR_Eigvals(A, iterations=500000):

    A_old = np.copy(A)
    A_new = np.copy(A)

    i = 0
    while i < iterations:
        A_old = A_new
        Q, R = QR_Decomp(A_old)

        A_new = R @ Q

        i += 1

    subA = A_new[[0,1],:][:,[0,1]]

    lambda_3 = (subA[0][0]+subA[1][1])/2 + cmath.sqrt((subA[0][0]+subA[1][1])**2/4 -subA[0][0]*subA[1][1]+subA[0][1]*subA[1][0])
    lambda_4 = (subA[0][0]+subA[1][1])/2 - cmath.sqrt((subA[0][0]+subA[1][1])**2/4 -subA[0][0]*subA[1][1]+subA[0][1]*subA[1][0])

    eigvals = np.array([A_new[2][2], A_new[3][3], lambda_3, lambda_4])

    return eigvals
