import numpy as np
import scipy as sci
import math
import cmath
import DDS as dds
import prettyprint as pretty
import matplotlib.pyplot as plt


def Gamma_m_n(m, n, Q_DVR, H_DVR, omega, gamma, coupling, temp):

    Y = -(Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

    W = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*coupling**2/omega**2

    A = (Q_DVR[m][m]-Q_DVR[n][n])**2 * gamma/2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

    B = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 8*gamma*coupling**2/omega**2 *temp

    C = -(Q_DVR[m][m]-Q_DVR[n][n])**2 * 2*gamma*coupling**2/omega**3 * (omega/temp + 2*np.sinh(omega/temp))/(np.cosh(omega/temp) - 1)

    V = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*gamma*coupling**2/omega**3

    E = H_DVR[m,m]-H_DVR[n,n]

    t = np.linspace(0,2000,200000)

    Kernel_m_n = (2*H_DVR[m][n])**2/2 * np.exp( -Y*np.cos(omega*t) +Y -A*t*np.cos(omega*t) -B*t -C*np.sin(omega*t) ) * np.cos( E*t +W*np.sin(omega*t) +V -V*np.cos(omega*t) -V*omega/2*np.sin(omega*t)  )

    val = np.trapz(Kernel_m_n, t)

    return val


def Gamma(n_sys, barrier, bias, omega, gamma, coupling, temp):

    H_DVR, Q_DVR = dds.calc_H_DVR_Q_DVR(n_sys, barrier, bias)[0], dds.calc_H_DVR_Q_DVR(n_sys, barrier, bias)[1]

    mat =  np.empty((4**n_sys,4**n_sys))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if j!=i:
                mat[i, j] = Gamma_m_n(i, j, Q_DVR, H_DVR, omega, gamma, coupling, temp)
            if j==i:
                mat[i, j] = 0

    for i in range(mat.shape[0]):
        mat[i,i] = -np.sum(mat[:,i])

    return mat


print(Gamma(2, 1.4, 0.05, 1, 0.097, 0.18, 0.5))
