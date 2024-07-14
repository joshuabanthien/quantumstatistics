import numpy as np
import scipy as sci
import math
import cmath
import DDS as dds
import prettyprint as pretty
import matplotlib.pyplot as plt

H_DVR, Q_mu = dds.calc_H_DVR_Q_mu(1.4, 0.05)

def H_mu_nu(m, n, tau, omega, gamma, coupling, temp):

    Y = -(Q_mu[m]-Q_mu[n])**2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

    W = (Q_mu[m]-Q_mu[n])**2 * 4*coupling**2/omega**2

    A = (Q_mu[m]-Q_mu[n])**2 * gamma/2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

    B = (Q_mu[m]-Q_mu[n])**2 * 8*gamma*coupling**2/omega**3 *temp

    C = -(Q_mu[m]-Q_mu[n])**2 * 2*gamma*coupling**2/omega**3 * (omega/temp + 2*np.sinh(omega/temp))/(np.cosh(omega/temp) - 1)

    V = (Q_mu[m]-Q_mu[n])**2 * 4*gamma*coupling**2/omega**3

    E = H_DVR[m,m]-H_DVR[n,n]

    u_0 = complex(-V,Y)

    v_0 = complex(W,C)

    w_0 = complex(V,Y)

    value = H_DVR[m,n]**2 /2 * np.exp( -Y*(np.cos(omega*tau) -1) -A*tau*np.cos(omega*tau) -B*tau -C*np.sin(omega*tau) )*np.cos( E*tau + W*np.sin(omega*tau) +V -V*np.cos(omega*tau) - V*omega/2*tau*np.sin(omega*tau) )

    return value


def H_mu_nu_approx(m, n, tau, omega, gamma, coupling, temp):

    Y = -(Q_mu[m]-Q_mu[n])**2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

    W = (Q_mu[m]-Q_mu[n])**2 * 4*coupling**2/omega**2

    A = (Q_mu[m]-Q_mu[n])**2 * gamma/2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

    B = (Q_mu[m]-Q_mu[n])**2 * 8*gamma*coupling**2/omega**3 *temp

    C = -(Q_mu[m]-Q_mu[n])**2 * 2*gamma*coupling**2/omega**3 * (omega/temp + 2*np.sinh(omega/temp))/(np.cosh(omega/temp) - 1)

    V = (Q_mu[m]-Q_mu[n])**2 * 4*gamma*coupling**2/omega**3

    E = H_DVR[m,m]-H_DVR[n,n]

    u_0 = complex(0,1)*np.sqrt(Y**2 - W**2)

    valneg1 = complex(0,-1)**(-1) * sci.special.jv(-1,u_0) * np.exp(-omega/(2*temp)) * ( np.cos(E*tau-omega*tau) - A*tau*np.cos(E*tau-omega*tau)*np.cos(omega*tau) \
                + V*omega/2 *np.sin(E*tau-omega*tau)*np.sin(omega*tau) - C*np.cos(E*tau-omega*tau)*np.sin(omega*tau) + V*np.sin(E*tau-omega*tau)*np.cos(omega*tau) \
                + V*np.sin(E*tau-omega*tau) )

    val0 = sci.special.jv(0,u_0) * ( np.cos(E*tau) - A*tau*np.cos(E*tau)*np.cos(omega*tau) \
                + V*omega/2 *np.sin(E*tau)*np.sin(omega*tau) - C*np.cos(E*tau)*np.sin(omega*tau) + V*np.sin(E*tau)*np.cos(omega*tau) \
                + V*np.sin(E*tau) )

    valpos1 = complex(0,-1) * sci.special.jv(1,u_0) * np.exp(omega/(2*temp)) * ( np.cos(E*tau+omega*tau) - A*tau*np.cos(E*tau+omega*tau)*np.cos(omega*tau) \
                + V*omega/2 *np.sin(E*tau+omega*tau)*np.sin(omega*tau) - C*np.cos(E*tau+omega*tau)*np.sin(omega*tau) + V*np.sin(E*tau+omega*tau)*np.cos(omega*tau) \
                + V*np.sin(E*tau+omega*tau) )

    value = H_DVR[m,n]**2 /2 * np.exp(Y) * np.exp(-B*tau) * ( valneg1 + val0 + valpos1 )

    return value


x = np.linspace(0,100,500)

H_mu_nu_approx_vec = np.vectorize(H_mu_nu_approx)
H_mu_nu_vec = np.vectorize(H_mu_nu)

plt.plot(x,H_mu_nu_vec(0, 2, x, 1, 0.097, 0.18, 0.5), label='true')
plt.plot(x,H_mu_nu_approx_vec(0, 2, x, 1, 0.097, 0.18, 0.5), label='approx')
plt.legend()
plt.show()
