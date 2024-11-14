import numpy as np
import sim_density_matrix
import rate_matrix
import math_collection
import dds
import matplotlib.pyplot as plt
import numerical_fourier_transform

n_sys = 1
barrier = 1.4
bias = 0.05
omega = 1
gamma = 0.097
coupling = 0.18
temp = 1
drive_coup = 0.4
drive_freq = 0.8
t_end = 20
n_steps = 5000


def LINEAR_RESPONSE_FUNCTION(RHO_EQUI, RHO_S, Q_DVR, T):

    ex_value_evo = Q_DVR[0,0]*RHO_S[:,0]+Q_DVR[1,1]*RHO_S[:,1]+Q_DVR[2,2]*RHO_S[:,2]+Q_DVR[3,3]*RHO_S[:,3] - (Q_DVR[0,0]*RHO_EQUI[0]+Q_DVR[1,1]*RHO_EQUI[1]+Q_DVR[2,2]*RHO_EQUI[2]+Q_DVR[3,3]*RHO_EQUI[3])

    #val = complex(0,1)*ex_value_evo

    val = ex_value_evo

    return val


#def linear_absorption_spectrum(n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq, t_end, n_steps):
#
#        x_in = np.linspace(0,t_end,n_steps)
#
#        T = np.linspace(0,30,5000)
#
#        LIN_RESPONSE_VALUES = linear_response_function(n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq, t_end, n_steps)
#
#        INTEGRAL = numerical_fourier_transform.fourier_transform_data_imag(T,x_in,LIN_RESPONSE_VALUES)
#
#        return INTEGRAL


#def linear_absorption_integrand(n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq, t_end, n_steps):
#
#        x_in = np.linspace(0,t_end,n_steps)
#
#        LIN_RESPONSE_VALUES = linear_response_function(n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq, t_end, n_steps)
#
#        INTEGRAND = numerical_fourier_transform.fourier_transform_imag_integrand(9*np.pi,x_in,LIN_RESPONSE_VALUES)
#
#        return INTEGRAND




#print(linear_absorption_spectrum(n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq, t_end, n_steps))

#X_VALUES = np.linspace(0,t_end,n_steps)
#Y_VALUES = linear_absorption_integrand(n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq, t_end, n_steps)

#plt.plot(X_VALUES,Y_VALUES)
#plt.show()
