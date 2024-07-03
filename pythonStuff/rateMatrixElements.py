import numpy as np
import scipy as sci
import math
import cmath

g = 2
Omega = 11
Gamma = 0.001
T = 10
E_mu = [-0.1710061, -0.46948549, -0.61633741, -0.83104127]
Q_mu = [-3.386, -1.548, 2.004, 3.616]
H_DVR = np.array([[-0.61633741,  0.30726346, -0.02322276,  0.01349684],
                  [ 0.30726346, -0.1710061,  -0.05663543,  0.02801363],
                  [-0.02322276, -0.05663543, -0.46948549, -0.39096136],
                  [ 0.01349684,  0.02801363, -0.39096136, -0.83104127]])

def delta(m,n):
    return H_DVR[m][n]

def delE(m,n):
    return (E_mu[m]-E_mu[n])

def delQ(m,n):
    return Q_mu[m]-Q_mu[n]

def Y(m,n):
    return -delQ(m,n)**2 *4*g**2/Omega**2 * 1/math.tanh(Omega/(2*T))

def W(m,n):
    return delQ(m,n)*4*g**2/Omega**2

def A(m,n):
    return delQ(m,n)*Gamma/2 *4*g**2/Omega**2 * 1/math.tanh(Omega/(2*T))

def B(m,n):
    return delQ(m,n)*Gamma*8*g**2/Omega**3 *T

def C(m,n):
    return -delQ(m,n)*Gamma*2*g**2/Omega**3 *(Omega/T+2*math.sinh(Omega/T))/(math.cosh(Omega/T)-1)

def V(m,n):
    return delQ(m,n)*Gamma*4*g**2/Omega**3

def u(m,n):
    return complex(-V(m,n), -Y(m,n))

def v(m,n):
    z= complex( W(m,n), C(m,n))
    return z

def w(m,n):
    return A(m,n)/2 + Omega*V(m,n)/4


def sum_0_0(m,n):
    return sci.special.jv(0,-V(m,n)-Y(m,n)*j)*sci.special.jv(0,v(m,n))*( 1/complex(B(m,n),-delE(m,n)) + w(m,n)/complex(B(m,n),-delE(m,n)-Omega)**2+ w(m,n)/complex(B(m,n),-delE(m,n)+Omega)**2 )

def sum_1_0(m,n):
    return complex(0,1)*sci.special.jv(1,u(m,n))*sci.special.jv(0,v(m,n))*( 1/complex(B(m,n),-delE(m,n)-Omega) + w(m,n)/complex(B(m,n),-delE(m,n))**2 )

def sum_0_1(m,n):
    return sci.special.jv(0,u(m,n))*sci.special.jv(1,v(m,n))*( 1/complex(B(m,n),-delE(m,n)-Omega) + w(m,n)/complex(B(m,n),-delE(m,n))**2 )

def sum_neg1_0(m,n):
    return complex(0,-1)*sci.special.jv(-1,u(m,n))*sci.special.jv(0,v(m,n))*( 1/complex(B(m,n),-delE(m,n)+Omega) + w(m,n)/complex(B(m,n),-delE(m,n))**2 )

def sum_0_neg1(m,n):
    return sci.special.jv(0,u(m,n))*sci.special.jv(-1,v(m,n))*( 1/complex(B(m,n),-delE(m,n)+Omega) + w(m,n)/complex(B(m,n),-delE(m,n))**2 )

Gamma_mu_nu =  np.empty((4,4))

for i in range(Gamma_mu_nu.shape[0]):
    for j in range(Gamma_mu_nu.shape[1]):
        if j!=i:
            Gamma_mu_nu[i, j] = delta(i,j)**2/2*(np.exp(complex(-V(i,j),-Y(i,j)))*( sum_0_0(i,j)+sum_1_0(i,j)+sum_0_1(i,j)+sum_neg1_0(i,j)+sum_0_neg1(i,j))).real
        if j==i:
            Gamma_mu_nu[i, j] = 0

Gamma_mu_nu[0][0] = - ( Gamma_mu_nu[1][0] + Gamma_mu_nu[2][0] + Gamma_mu_nu[3][0] )
Gamma_mu_nu[1][1] = - ( Gamma_mu_nu[0][1] + Gamma_mu_nu[2][1] + Gamma_mu_nu[3][1] )
Gamma_mu_nu[2][2] = - ( Gamma_mu_nu[0][2] + Gamma_mu_nu[1][2] + Gamma_mu_nu[3][2] )
Gamma_mu_nu[3][3] = - ( Gamma_mu_nu[0][3] + Gamma_mu_nu[1][3] + Gamma_mu_nu[2][3] )

print(sorted(sci.linalg.eigvals(Gamma_mu_nu)))
print(Gamma_mu_nu)
