import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

Delta = 1.3e10
Omega = 1
Gamma = 0.097
g = 0.18
alpha = 4e-3
T = 0.1
hbar = 1.05e-34
k_B = 1.38e-23
beta = 1/(k_B*T)

Y = -4*g**2/Omega**2 * 1/np.tanh(Delta*beta*hbar*Omega/2)
W = 4*g**2/Omega**2
A = -Delta*Gamma*Y/2
B = Gamma*8*g**2/(Omega**3 * hbar*beta)
C = -Gamma*2*g**2/Omega**3 * (Delta*beta*hbar*Omega + 2*np.sinh(Delta*beta*hbar*Omega))/(np.cosh(Delta*beta*hbar*Omega) -1)
V = Gamma*4*g**2/Omega**3

def S(t):
    return Y*(np.cos(Omega*t)-1) + A*t/Delta*np.cos(Omega*t) + B*t/Delta + C*np.sin(Omega*t)

def R(t):
    return W*np.sin(Omega*t) + V*(1 - np.cos(Omega*t) - Omega/2 *t/Delta*np.sin(Omega*t))

def kernel(t):
    return np.exp(-S(t))*np.cos(R(t))
 \
def kernel1(t):
    return np.exp(-Y*(np.cos(Omega*t)-1))*( (np.cos(W*np.sin(Omega*t)))*(1-A*t/Delta*np.cos(Omega*t) - B*t/Delta - C*np.sin(Omega*t)) -np.sin(W*np.sin(Omega*t))*(V*(1 - np.cos(Omega*t) - Omega/2 *t/Delta*np.sin(Omega*t))) )

x = np.linspace(0, 5, 8000)

print(scipy.integrate.trapezoid(kernel(x), x))
print(scipy.integrate.trapezoid(kernel1(x), x))

print(1/(beta*hbar*Omega))

#plt.plot(x, np.exp(-S(x)))
#plt.plot(x, np.cos(x*Delta + R(x)))
plt.plot(x, kernel(x), label='0')
plt.plot(x, kernel1(x), label='1')
plt.legend()
plt.show()
