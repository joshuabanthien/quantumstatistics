import numpy as np
import rate_matrix
import dds
import matplotlib.pyplot as plt
import runge_kutta


params = {'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath} \usepackage{braket}'}
plt.rcParams.update(params)


def f(y,t,M):

    return M @ y


def DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX, SIM_TIME):

    DENSITY_MATRIX_SIM = runge_kutta.runge_kutta_1st_order(f, SIM_TIME, RHO_INIT, RATE_MATRIX)

    return SIM_TIME, DENSITY_MATRIX_SIM


def P_L(DENSITY_MATRIX_SIM, SIM_TIME, Q_DVR):

    P_L_SIM = np.zeros(DENSITY_MATRIX_SIM.shape[0])

    for i in range(DENSITY_MATRIX_SIM.shape[1]):

        if Q_DVR[i,i] < 0:

            P_L_SIM += DENSITY_MATRIX_SIM[:,i]

    return SIM_TIME, P_L_SIM


def P_R(DENSITY_MATRIX_SIM, SIM_TIME, Q_DVR):

    P_R_SIM = np.zeros(DENSITY_MATRIX_SIM.shape[0])

    for i in range(DENSITY_MATRIX_SIM.shape[1]):

        if Q_DVR[i,i] > 0:

            P_R_SIM += DENSITY_MATRIX_SIM[:,i]

    return SIM_TIME, P_R_SIM


n_sys = 1
barrier = 1.4
bias = 0.05
OMEGA = [0.3,0.55,0.8,1,1.2,10,20]
gamma = 0.097
COUPLING = [0,0.025,0.05,0.18,0.2,0.3,1]
TEMP = [0.3,0.65,1,2.5,5,10,20]

RHO_INIT = np.empty(4**n_sys)
RHO_INIT[0] = 1

SIM_TIME = np.linspace(0,5000,5000)
SHORT_SIM_TIME = np.linspace(0,200,5000)

H_DVR, Q_DVR = dds.H_DVR_Q_DVR(n_sys, barrier, bias)[0], dds.H_DVR_Q_DVR(n_sys, barrier, bias)[1]


COLORS = ['red', 'blue', 'green', 'black', 'violet', 'yellow', 'orange', 'turquoise']


###########################################

fig1, ax1 = plt.subplots()

ax1.set_xlim(0,5000)
ax1.set_ylim(0,1.1)
ax1.set_xlabel(r"$t$ in units of $\omega_0^{-1}$")
ax1.set_ylabel(r"$P_\text{L}$")

for coupling, color in zip(COUPLING, COLORS):
    RATE_MATRIX_COUP = rate_matrix.RATE_MATRIX(H_DVR, Q_DVR, OMEGA[3], gamma, coupling, TEMP[2])
    DENSITY_MATRIX_SIM_COUP = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_COUP, SIM_TIME)[1]
    P_L_SIM_COUP = P_L(DENSITY_MATRIX_SIM_COUP, SIM_TIME, Q_DVR)[1]
    ax1.plot(SIM_TIME,P_L_SIM_COUP, color=color, label=r"$g={}$".format(coupling))

ax1.text(3000,0.15, r'$E_\text{b}=$'+str(barrier))
ax1.text(3000,0.1, r'$\epsilon=$'+str(bias))
ax1.text(3000,0.05, r'$\Omega=$'+str(OMEGA[3]))
ax1.text(4000,0.15, r'$\Gamma=$'+str(gamma))
ax1.text(4000,0.1, r'$T=$'+str(TEMP[2]))
ax1.set_aspect(5000/1.1)
ax1.legend(frameon=False)
fig1.savefig('P_L_COUP.png', dpi=300, bbox_inches='tight')


###########################################

fig2, ax2 = plt.subplots()

ax2.set_xlim(0,5000)
ax2.set_ylim(0,1.1)
ax2.set_xlabel(r"$t$ in units of $\omega_0^{-1}$")
ax2.set_ylabel(r"$P_\text{L}$")

for omega, color in zip(OMEGA, COLORS):
    RATE_MATRIX_OMEGA = rate_matrix.RATE_MATRIX(H_DVR, Q_DVR, omega, gamma, COUPLING[3], TEMP[2])
    DENSITY_MATRIX_SIM_OMEGA = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_OMEGA, SIM_TIME)[1]
    P_L_SIM_OMEGA = P_L(DENSITY_MATRIX_SIM_OMEGA, SIM_TIME, Q_DVR)[1]
    ax2.plot(SIM_TIME,P_L_SIM_OMEGA, color=color, label=r"$\Omega={}$".format(omega))

ax2.text(3000,0.15, r'$E_\text{b}=$'+str(barrier))
ax2.text(3000,0.1, r'$\epsilon=$'+str(bias))
ax2.text(3000,0.05, r'$g=$'+str(COUPLING[3]))
ax2.text(4000,0.15, r'$\Gamma=$'+str(gamma))
ax2.text(4000,0.1, r'$T=$'+str(TEMP[2]))
ax2.set_aspect(5000/1.1)
ax2.legend(frameon=False)
fig2.savefig('P_L_OMEGA.png', dpi=300, bbox_inches='tight')


###########################################

fig3, ax3 = plt.subplots()

fig3_omega = OMEGA[3]
fig3_coupling = COUPLING[6]
fig3_temp = TEMP[2]

ax3.set_xlim(0,5000)
ax3.set_ylim(0,1.1)
ax3.set_xlabel(r"$t$ in units of $\omega_0^{-1}$")
#ax3.set_ylabel(r"$P_\text{L}$")
RATE_MATRIX_DENS_MAT = rate_matrix.RATE_MATRIX(H_DVR, Q_DVR, fig3_omega, gamma, fig3_coupling, fig3_temp)
DENSITY_MATRIX_SIM_1 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SIM_TIME)[1][:,0]
DENSITY_MATRIX_SIM_2 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SIM_TIME)[1][:,1]
DENSITY_MATRIX_SIM_3 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SIM_TIME)[1][:,2]
DENSITY_MATRIX_SIM_4 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SIM_TIME)[1][:,3]
ax3.plot(SIM_TIME, DENSITY_MATRIX_SIM_1, color=COLORS[0], label=r'$\rho_{11}$')
ax3.plot(SIM_TIME, DENSITY_MATRIX_SIM_2, color=COLORS[1], label=r'$\rho_{22}$')
ax3.plot(SIM_TIME, DENSITY_MATRIX_SIM_3, color=COLORS[2], label=r'$\rho_{33}$')
ax3.plot(SIM_TIME, DENSITY_MATRIX_SIM_4, color=COLORS[3], label=r'$\rho_{44}$')
ax3.text(1000,1, r'$E_\text{b}=$'+str(barrier))
ax3.text(1000,0.95, r'$\epsilon=$'+str(bias))
ax3.text(1000,0.9, r'$g=$'+str(fig3_coupling))
ax3.text(2000,1, r'$\Gamma=$'+str(gamma))
ax3.text(2000,0.95, r'$T=$'+str(fig3_temp))
ax3.text(2000,0.9, r'$\Omega=$'+str(fig3_omega))
ax3.set_aspect(5000/1.1)
ax3.legend(frameon=False)
fig3.savefig('DENS_MAT_LONG.png', dpi=300, bbox_inches='tight')


fig4, ax4 = plt.subplots()

ax4.set_xlim(0,200)
ax4.set_ylim(0,1.1)
ax4.set_xlabel(r"$t$ in units of $\omega_0^{-1}$")
#ax3.set_ylabel(r"$P_\text{L}$")
RATE_MATRIX_DENS_MAT = rate_matrix.RATE_MATRIX(H_DVR, Q_DVR, fig3_omega, gamma, fig3_coupling, fig3_temp)
DENSITY_MATRIX_SIM_1 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SHORT_SIM_TIME)[1][:,0]
DENSITY_MATRIX_SIM_2 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SHORT_SIM_TIME)[1][:,1]
DENSITY_MATRIX_SIM_3 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SHORT_SIM_TIME)[1][:,2]
DENSITY_MATRIX_SIM_4 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SHORT_SIM_TIME)[1][:,3]
ax4.plot(SHORT_SIM_TIME, DENSITY_MATRIX_SIM_1, color=COLORS[0], label=r'$\rho_{11}$')
ax4.plot(SHORT_SIM_TIME, DENSITY_MATRIX_SIM_2, color=COLORS[1], label=r'$\rho_{22}$')
ax4.plot(SHORT_SIM_TIME, DENSITY_MATRIX_SIM_3, color=COLORS[2], label=r'$\rho_{33}$')
ax4.plot(SHORT_SIM_TIME, DENSITY_MATRIX_SIM_4, color=COLORS[3], label=r'$\rho_{44}$')
ax4.text(50,1, r'$E_\text{b}=$'+str(barrier))
ax4.text(50,0.95, r'$\epsilon=$'+str(bias))
ax4.text(50,0.9, r'$g=$'+str(fig3_coupling))
ax4.text(100,1, r'$\Gamma=$'+str(gamma))
ax4.text(100,0.95, r'$T=$'+str(fig3_temp))
ax4.text(100,0.9, r'$\Omega=$'+str(fig3_omega))
ax4.set_aspect(200/1.1)
ax4.legend(frameon=False)
fig4.savefig('DENS_MAT_SHORT.png', dpi=300, bbox_inches='tight')


###########################################

fig5, ax5 = plt.subplots()

ax5.set_xlim(0,5000)
ax5.set_ylim(0,1.1)
ax5.set_xlabel(r"$t$ in units of $\omega_0^{-1}$")
ax5.set_ylabel(r"$P_\text{L}$")

for temp, color in zip(TEMP, COLORS):
    RATE_MATRIX_TEMP = rate_matrix.RATE_MATRIX(H_DVR, Q_DVR, OMEGA[3], gamma, COUPLING[3], temp)
    DENSITY_MATRIX_SIM_TEMP = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_TEMP, SIM_TIME)[1]
    P_L_SIM_TEMP = P_L(DENSITY_MATRIX_SIM_TEMP, SIM_TIME, Q_DVR)[1]
    ax5.plot(SIM_TIME,P_L_SIM_TEMP, color=color, label=r"$T={}$".format(temp))

ax5.text(3000,0.15, r'$E_\text{b}=$'+str(barrier))
ax5.text(3000,0.1, r'$\epsilon=$'+str(bias))
ax5.text(3000,0.05, r'$g=$'+str(COUPLING[3]))
ax5.text(4000,0.15, r'$\Gamma=$'+str(gamma))
ax5.text(4000,0.1, r'$\Omega=$'+str(OMEGA[3]))
ax5.set_aspect(5000/1.1)
ax5.legend(frameon=False)
fig5.savefig('P_L_TEMP.png', dpi=300, bbox_inches='tight')


###########################################

fig6, ax6 = plt.subplots()

fig6_omega = OMEGA[3]
fig6_coupling = COUPLING[3]
fig6_temp = TEMP[5]

ax6.set_xlim(0,5000)
ax6.set_ylim(0,1.1)
ax6.set_xlabel(r"$t$ in units of $\omega_0^{-1}$")
#ax3.set_ylabel(r"$P_\text{L}$")
RATE_MATRIX_DENS_MAT = rate_matrix.RATE_MATRIX(H_DVR, Q_DVR, fig6_omega, gamma, fig6_coupling, fig6_temp)
DENSITY_MATRIX_SIM_1 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SIM_TIME)[1][:,0]
DENSITY_MATRIX_SIM_2 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SIM_TIME)[1][:,1]
DENSITY_MATRIX_SIM_3 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SIM_TIME)[1][:,2]
DENSITY_MATRIX_SIM_4 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SIM_TIME)[1][:,3]
ax6.plot(SIM_TIME, DENSITY_MATRIX_SIM_1, color=COLORS[0], label=r'$\rho_{11}$')
ax6.plot(SIM_TIME, DENSITY_MATRIX_SIM_2, color=COLORS[1], label=r'$\rho_{22}$')
ax6.plot(SIM_TIME, DENSITY_MATRIX_SIM_3, color=COLORS[2], label=r'$\rho_{33}$')
ax6.plot(SIM_TIME, DENSITY_MATRIX_SIM_4, color=COLORS[3], label=r'$\rho_{44}$')
ax6.text(1000,1, r'$E_\text{b}=$'+str(barrier))
ax6.text(1000,0.95, r'$\epsilon=$'+str(bias))
ax6.text(1000,0.9, r'$g=$'+str(fig6_coupling))
ax6.text(2000,1, r'$\Gamma=$'+str(gamma))
ax6.text(2000,0.95, r'$T=$'+str(fig6_temp))
ax6.text(2000,0.9, r'$\Omega=$'+str(fig6_omega))
ax6.set_aspect(5000/1.1)
ax6.legend(frameon=False)
fig6.savefig('DENS_MAT_LONG_TEMP.png', dpi=300, bbox_inches='tight')


fig7, ax7 = plt.subplots()

ax7.set_xlim(0,200)
ax7.set_ylim(0,1.1)
ax7.set_xlabel(r"$t$ in units of $\omega_0^{-1}$")
#ax3.set_ylabel(r"$P_\text{L}$")
RATE_MATRIX_DENS_MAT = rate_matrix.RATE_MATRIX(H_DVR, Q_DVR, fig6_omega, gamma, fig6_coupling, fig6_temp)
DENSITY_MATRIX_SIM_1 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SHORT_SIM_TIME)[1][:,0]
DENSITY_MATRIX_SIM_2 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SHORT_SIM_TIME)[1][:,1]
DENSITY_MATRIX_SIM_3 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SHORT_SIM_TIME)[1][:,2]
DENSITY_MATRIX_SIM_4 = DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DENS_MAT, SHORT_SIM_TIME)[1][:,3]
ax7.plot(SHORT_SIM_TIME, DENSITY_MATRIX_SIM_1, color=COLORS[0], label=r'$\rho_{11}$')
ax7.plot(SHORT_SIM_TIME, DENSITY_MATRIX_SIM_2, color=COLORS[1], label=r'$\rho_{22}$')
ax7.plot(SHORT_SIM_TIME, DENSITY_MATRIX_SIM_3, color=COLORS[2], label=r'$\rho_{33}$')
ax7.plot(SHORT_SIM_TIME, DENSITY_MATRIX_SIM_4, color=COLORS[3], label=r'$\rho_{44}$')
ax7.text(50,1, r'$E_\text{b}=$'+str(barrier))
ax7.text(50,0.95, r'$\epsilon=$'+str(bias))
ax7.text(50,0.9, r'$g=$'+str(fig6_coupling))
ax7.text(100,1, r'$\Gamma=$'+str(gamma))
ax7.text(100,0.95, r'$T=$'+str(fig6_temp))
ax7.text(100,0.9, r'$\Omega=$'+str(fig6_omega))
ax7.set_aspect(200/1.1)
ax7.legend(frameon=False)
fig7.savefig('DENS_MAT_SHORT_TEMP.png', dpi=300, bbox_inches='tight')

gc.collect()
