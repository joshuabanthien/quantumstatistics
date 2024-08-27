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

    u_0 = complex(0,1)*np.sqrt(Y**2 - W**2)

    val = 0


    if H_DVR[m,n] != 0:

        val1 = complex(0,1)*sci.special.jv(-1,u_0)*np.exp(-omega/(2*temp)) * ( (B-V*(E-omega))/(B**2+(E-omega)**2) - (A/2+V*omega/4)*(B**2-E**2)/(B**2+E**2)**2 - (C/2-V/2)*E/(B**2+E**2) )

        val2 = sci.special.jv(0,u_0) * ( B/(B**2+E**2) - (A/2-V*omega/4)*(B**2-(E-omega)**2)/(B**2+(E-omega)**2)**2 - (A/2+V*omega/4)*(B**2-(E+omega)**2)/(B**2+(E+omega)**2)**2 \
                + (C/2+V/2)*(E-omega)/(B**2+(E-omega)**2) + (V/2-C/2)*(E+omega)/(B**2+(E+omega)**2) - V*E/(B**2+E**2) )

        val3 = complex(0,-1)*sci.special.jv(1,u_0)*np.exp(omega/(2*temp)) * ( (B-V*(E+omega))/(B**2+(E+omega)**2) - (A/2-V*omega/4)*(B**2-E**2)/(B**2+E**2)**2 + (C/2+V/2)*E/(B**2+E**2) )

        val = (2*H_DVR[m,n])**2/2 * np.exp(Y)* (val1+val2+val3)


    return val


def Gamma(n_sys, barrier, bias, omega, gamma, coupling, temp):

    H_DVR, Q_DVR = dds.calc_H_DVR_Q_mu(n_sys, barrier, bias)[0], dds.calc_H_DVR_Q_mu(n_sys, barrier, bias)[1]

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


def Gamma_eigvals(n_sys, barrier, bias, omega, gamma, coupling, temp):

    mat = Gamma(n_sys, barrier, bias, omega, gamma, coupling, temp)

    vals = np.linalg.eigvals(mat)

    return vals


def smallest_nonzero_eigval(n_sys, barrier, bias, omega, gamma, coupling, temp):

    mat = Gamma(n_sys, barrier, bias, omega, gamma, coupling, temp)

    eigvals = np.abs(sorted(np.real(np.linalg.eigvals(mat))))

    val = min(i for i in eigvals if i > 1e-10)

    return val
