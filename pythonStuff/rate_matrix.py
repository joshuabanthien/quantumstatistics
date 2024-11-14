import numpy as np
import scipy as sci
import math
import cmath
import dds
import prettyprint as pretty
import matplotlib.pyplot as plt


#calculates the rate matrix via the approximated matrix elements

def RATE_MATRIX(H_DVR, Q_DVR, omega, gamma, coupling, temp):

    n = len(range(H_DVR.shape[0]))

    M =  np.empty((n,n))

    for m in range(M.shape[0]):

        for n in range(M.shape[1]):

            if n!=m:

                Y = -(Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

                W = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*coupling**2/omega**2

                A = (Q_DVR[m][m]-Q_DVR[n][n])**2 * gamma/2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

                B = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 8*gamma*coupling**2/omega**2 *temp

                C = -(Q_DVR[m][m]-Q_DVR[n][n])**2 * 2*gamma*coupling**2/omega**3 * (omega/temp + 2*np.sinh(omega/temp))/(np.cosh(omega/temp) - 1)

                V = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*gamma*coupling**2/omega**3

                E = H_DVR[m,m]-H_DVR[n,n]

                u_0 = complex(0,1)*np.sqrt(Y**2 - W**2)

                if H_DVR[m,n] != 0:

                    val1 = complex(0,1)*sci.special.jv(-1,u_0)*np.exp(-omega/(2*temp)) * ( (B-V*(E-omega))/(B**2+(E-omega)**2) - (A/2+V*omega/4)*(B**2-E**2)/(B**2+E**2)**2 - (C/2-V/2)*E/(B**2+E**2) )

                    val2 = sci.special.jv(0,u_0) * ( B/(B**2+E**2) - (A/2-V*omega/4)*(B**2-(E-omega)**2)/(B**2+(E-omega)**2)**2 - (A/2+V*omega/4)*(B**2-(E+omega)**2)/(B**2+(E+omega)**2)**2 \
                            + (C/2+V/2)*(E-omega)/(B**2+(E-omega)**2) + (V/2-C/2)*(E+omega)/(B**2+(E+omega)**2) - V*E/(B**2+E**2) )

                    val3 = complex(0,-1)*sci.special.jv(1,u_0)*np.exp(omega/(2*temp)) * ( (B-V*(E+omega))/(B**2+(E+omega)**2) - (A/2-V*omega/4)*(B**2-E**2)/(B**2+E**2)**2 + (C/2+V/2)*E/(B**2+E**2) )

                    M[m,n] = (2*H_DVR[m,n])**2/2 * np.exp(Y)* (val1+val2+val3)

                if H_DVR[m,n] == 0:

                    M[m,n] = 0

            if n==m:

                M[m,n] = 0

    for i in range(M.shape[0]):

        M[i,i] = -np.sum(M[:,i])

    return M


#calculates the rate matrix via numerical integration, without driving

def RATE_MATRIX_UNAPPROX(H_DVR, Q_DVR, omega, gamma, coupling, temp):

    n = len(range(H_DVR.shape[0]))

    M =  np.empty((n,n))

    for m in range(M.shape[0]):

        for n in range(M.shape[1]):

            if n!=m:

                Y = -(Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

                W = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*coupling**2/omega**2

                A = (Q_DVR[m][m]-Q_DVR[n][n])**2 * gamma/2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

                B = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 8*gamma*coupling**2/omega**2 *temp

                C = -(Q_DVR[m][m]-Q_DVR[n][n])**2 * 2*gamma*coupling**2/omega**3 * (omega/temp + 2*np.sinh(omega/temp))/(np.cosh(omega/temp) - 1)

                V = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*gamma*coupling**2/omega**3

                E = H_DVR[m,m]-H_DVR[n,n]

                X_INT = np.linspace(0,100,1000)

                if H_DVR[m,n] != 0:

                    Y_VALUES = (2*H_DVR[m,n])**2/2*np.exp(-(Y*(np.cos(omega*X_INT)-1)+A*X_INT*np.cos(omega*X_INT)+B*X_INT+C*np.sin(omega*X_INT)))*np.cos(E*X_INT+W*np.sin(omega*X_INT)+V*(1-np.cos(omega*X_INT)-omega/2*X_INT*np.sin(omega*X_INT)))

                    M[m,n] = np.trapz(Y_VALUES,X_INT)

                if H_DVR[m,n] == 0:

                    M[m,n] = 0

            if n==m:

                M[m,n] = 0

    for i in range(M.shape[0]):

        M[i,i] = -np.sum(M[:,i])

    return M


#calculates the rate matrix via numerical integration, with driving

def RATE_MATRIX_DRIVEN_UNAPPROX(H_DVR, Q_DVR, omega, gamma, coupling, temp, drive_coup, drive_freq):

    n = len(range(H_DVR.shape[0]))

    M =  np.empty((n,n))

    for m in range(M.shape[0]):

        for n in range(M.shape[1]):

            if n!=m:

                Y = -(Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

                W = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*coupling**2/omega**2

                A = (Q_DVR[m][m]-Q_DVR[n][n])**2 * gamma/2 * 4*coupling**2/omega**2 * 1/np.tanh(omega/(2*temp))

                B = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 8*gamma*coupling**2/omega**2 *temp

                C = -(Q_DVR[m][m]-Q_DVR[n][n])**2 * 2*gamma*coupling**2/omega**3 * (omega/temp + 2*np.sinh(omega/temp))/(np.cosh(omega/temp) - 1)

                V = (Q_DVR[m][m]-Q_DVR[n][n])**2 * 4*gamma*coupling**2/omega**3

                E = H_DVR[m,m]-H_DVR[n,n]

                X_INT = np.linspace(0,100,1000)

                if H_DVR[m,n] != 0:

                    Y_VALUES = (2*H_DVR[m,n])**2/2*np.exp(-(Y*(np.cos(omega*X_INT)-1)+A*X_INT*np.cos(omega*X_INT)+B*X_INT+C*np.sin(omega*X_INT)))*np.cos((E-(Q_DVR[m][m]-Q_DVR[m][m])*drive_coup*np.sin(drive_freq*X_INT))*X_INT+W*np.sin(omega*X_INT)+V*(1-np.cos(omega*X_INT)-omega/2*X_INT*np.sin(omega*X_INT)))*sci.special.jv(0,2*drive_coup/drive_freq*(Q_DVR[m][m]-Q_DVR[n][n])*np.sin(drive_freq/2*X_INT))

                    M[m,n] = np.trapz(Y_VALUES,X_INT)

                if H_DVR[m,n] == 0:

                    M[m,n] = 0

            if n==m:

                M[m,n] = 0

    for i in range(M.shape[0]):

        M[i,i] = -np.sum(M[:,i])

    return M
