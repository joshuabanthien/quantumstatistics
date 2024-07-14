import numpy as np
import scipy as sci
import math
import cmath
import DDS as dds
import prettyprint as pretty
import matplotlib.pyplot as plt


def Gamma_m_n(m, n, Q_mu, H_DVR, omega, gamma, coupling, temp):

    Y = -(Q_mu[m]-Q_mu[n])**2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

    W = (Q_mu[m]-Q_mu[n])**2 * 4*coupling**2/omega**2

    A = (Q_mu[m]-Q_mu[n])**2 * gamma/2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

    B = (Q_mu[m]-Q_mu[n])**2 * 8*gamma*coupling**2/omega**3 *temp

    C = -(Q_mu[m]-Q_mu[n])**2 * 2*gamma*coupling**2/omega**3 * (omega/temp + 2*np.sinh(omega/temp))/(np.cosh(omega/temp) - 1)

    V = (Q_mu[m]-Q_mu[n])**2 * 4*gamma*coupling**2/omega**3

    E = H_DVR[m,m]-H_DVR[n,n]

    u_0 = complex(-V,Y)

    v_0 = complex(W,C)

    w_0 = complex(Y,V)

    val1 = sci.special.jv(0,u_0)*sci.special.jv(0,v_0) * ( 1/complex(B,-E) - (A/2+V*omega/4)/complex(B,-E-omega)**2 - (A/2-V*omega/4)/complex(B,-E+omega)**2 )

    val2 = (sci.special.jv(0,u_0)*sci.special.jv(1,v_0)+complex(0,1)*sci.special.jv(1,u_0)*sci.special.jv(0,v_0)) * ( 1/complex(B,-E-omega) - (A/2-V*omega/4)/complex(B,-E)**2 )

    val3 = (sci.special.jv(0,u_0)*sci.special.jv(-1,v_0)+complex(0,-1)*sci.special.jv(-1,u_0)*sci.special.jv(0,v_0)) * ( 1/complex(B,-E+omega) - (A/2+V*omega/4)/complex(B,-E)**2 )

    val = H_DVR[m,n]**2/2 * np.real(np.exp(w_0)* (val1+val2+val3))

    return val


def Gamma(barrier, bias, omega, gamma, coupling, temp):

    H_DVR, Q_mu = dds.calc_H_DVR_Q_mu(barrier, bias)[0], dds.calc_H_DVR_Q_mu(barrier, bias)[1]

    mat =  np.empty((4,4))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if j!=i:
                mat[i, j] = Gamma_m_n(i, j, Q_mu, H_DVR, omega, gamma, coupling, temp)
            if j==i:
                mat[i, j] = 0

    mat[0][0] = - ( mat[1][0] + mat[2][0] + mat[3][0] )
    mat[1][1] = - ( mat[0][1] + mat[2][1] + mat[3][1] )
    mat[2][2] = - ( mat[0][2] + mat[1][2] + mat[3][2] )
    mat[3][3] = - ( mat[0][3] + mat[1][3] + mat[2][3] )

    return mat


def Gamma_eigvals(barrier, bias, omega, gamma, coupling, temp):

    mat = Gamma(barrier, bias, omega, gamma, coupling, temp)

    vals = sci.linalg.eigvals(mat)

    return vals


def smallest_nonzero_eigval(barrier, bias, omega, gamma, coupling, temp):

    mat = Gamma(barrier, bias, omega, gamma, coupling, temp)

    val = np.abs(sorted(np.real(sci.linalg.eigvals(mat))))[2]

    return val


x = np.linspace(0,0.2,200)

smallest_nonzero_eigval_vec = np.vectorize(smallest_nonzero_eigval)

plt.plot(x,smallest_nonzero_eigval_vec(1.4, x, 1, 0.097, 0.18, 0.5))

plt.show()
