import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import math
import dds


params = {'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath} \usepackage{braket}'}
plt.rcParams.update(params)


#H_DVR_BASIS, Q_DVR_BASIS, E_N, TRAFMAT_SHO_ENERGY, TRAFMAT_ENERGY_LOC, TRAFMAT_LOC_DVR = dds.H_DVR_Q_DVR(n_sys,E_b,epsilon)

K = 50

#function for plotting states of the harmonic oscillator

def SHO_N(n, x):

    return 1/np.sqrt(np.sqrt(np.pi) * 2**n *math.factorial(n)) * np.exp(-0.5*x**2) * sci.special.hermite(n, monic=False)(x)


#DDS energy eigenstates in position representation

def PSI_N(n, x, TRAFMAT_SHO_ENERGY):

    val = 0

    for i in range(0, K-1):

        val += TRAFMAT_SHO_ENERGY[n-1][i] * SHO_N(i,x)

    return val


#localized states

def L_1(x, TRAFMAT_SHO_ENERGY):

    return 1/np.sqrt(2) * (PSI_N(1, x, TRAFMAT_SHO_ENERGY)-PSI_N(2, x, TRAFMAT_SHO_ENERGY))


def R_1(x, TRAFMAT_SHO_ENERGY):

    return 1/np.sqrt(2) * (PSI_N(1, x, TRAFMAT_SHO_ENERGY)+PSI_N(2, x, TRAFMAT_SHO_ENERGY))


def L_2(x, TRAFMAT_SHO_ENERGY):

    return 1/np.sqrt(2) * (PSI_N(3, x, TRAFMAT_SHO_ENERGY)-PSI_N(4, x, TRAFMAT_SHO_ENERGY))


def R_2(x, TRAFMAT_SHO_ENERGY):
    return 1/np.sqrt(2) * (PSI_N(3, x, TRAFMAT_SHO_ENERGY)+PSI_N(4, x, TRAFMAT_SHO_ENERGY))


#DVR states

def Q_1(x, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR):

    return TRAFMAT_LOC_DVR[0][0]*L_1(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[0][1]*R_1(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[0][2]*L_2(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[0][3]*R_2(x, TRAFMAT_SHO_ENERGY)


def Q_2(x, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR):

    return TRAFMAT_LOC_DVR[1][0]*L_1(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[1][1]*R_1(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[1][2]*L_2(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[1][3]*R_2(x, TRAFMAT_SHO_ENERGY)


def Q_3(x, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR):

    return TRAFMAT_LOC_DVR[2][0]*L_1(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[2][1]*R_1(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[2][2]*L_2(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[2][3]*R_2(x, TRAFMAT_SHO_ENERGY)


def Q_4(x, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR):

    return TRAFMAT_LOC_DVR[3][0]*L_1(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[3][1]*R_1(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[3][2]*L_2(x, TRAFMAT_SHO_ENERGY) + TRAFMAT_LOC_DVR[3][3]*R_2(x, TRAFMAT_SHO_ENERGY)


def GENERATE_DDS_PLOTS(Q_DVR_BASIS, E_N, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR, barrier, bias):

    E_b = barrier

    epsilon = bias

    x=np.linspace(-6,6,600)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"$q$")
    ax1.set_ylabel(r"$V_0(q)$")
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-1, 2)
    ax1.set_xticks([-6,-4,-2,0,2,4,6])
    ax1.set_yticks([-0.8,-0.4,0,0.4,0.8,1.2,1.6,2])
    ax1.plot(x, 1/(64*E_b) *x**4 - 1/4 *x**2 +E_b - epsilon*x, color='k')
    ax1.plot(x, 1/2*PSI_N(1,x, TRAFMAT_SHO_ENERGY) + E_N[0] + E_b, "k--", linewidth=1, label=r'$\braket{q|\psi_1}$')
    ax1.plot(x, -1/2*PSI_N(2,x, TRAFMAT_SHO_ENERGY) + E_N[1] + E_b, "k:", linewidth=1, label=r'$\braket{q|\psi_2}$')
    ax1.plot(x, -1/2*PSI_N(3,x, TRAFMAT_SHO_ENERGY) + E_N[2] + E_b, "k-.", linewidth=1, label=r'$\braket{q|\psi_3}$')
    ax1.plot(x, 1/2*PSI_N(4,x, TRAFMAT_SHO_ENERGY) + E_N[3] + E_b, "k-", linewidth=1, label=r'$\braket{q|\psi_4}$')
    ax1.hlines(y= E_N[0] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
    ax1.hlines(y= E_N[1] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
    ax1.hlines(y= E_N[2] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
    ax1.hlines(y= E_N[3] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
    ax1.text(6.5, E_N[0] +E_b -0.05, r'$\mathcal{E}_1$')
    ax1.text(6.5, E_N[1] +E_b -0.05, r'$\mathcal{E}_2$')
    ax1.text(6.5, E_N[2] +E_b -0.05, r'$\mathcal{E}_3$')
    ax1.text(6.5, E_N[3] +E_b -0.05, r'$\mathcal{E}_4$')
    ax1.text(3, -0.7, r'$E_{\text{b}}=$'+str(barrier))
    ax1.text(3, -0.8, r'$\epsilon=$'+str(bias))
    ax1.set_aspect(5)
    ax1.legend(frameon=False)

    plt.savefig('DDS_ENERGY_BASIS.png', dpi=300, bbox_inches='tight')

    fig, ax2 = plt.subplots()

    ax2.set_xlabel(r"$q$"), TRAFMAT_SHO_ENERGY
    ax2.set_ylabel(r"$V_0(q)$")
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-1, 2)
    ax2.set_xticks([-6,-4,-2,0,2,4,6])
    ax2.set_yticks([-0.8,-0.4,0,0.4,0.8,1.2,1.6,2])
    ax2.plot(x, 1/(64*E_b) *x**4 - 1/4 *x**2 +E_b - epsilon*x, color='k')
    ax2.plot(x, 1/2*L_1(x, TRAFMAT_SHO_ENERGY) + 1/2*(E_N[0]+E_N[1]) + E_b, "k--", linewidth=1, label=r'$\braket{q|L_1}$')
    ax2.plot(x, -1/2*R_1(x, TRAFMAT_SHO_ENERGY) + 1/2*(E_N[0]+E_N[1]) + E_b, "k:", linewidth=1, label=r'$\braket{q|R_1}$')
    ax2.plot(x, 1/2*L_2(x, TRAFMAT_SHO_ENERGY) + 1/2*(E_N[2]+E_N[3]) + E_b, "k-.", linewidth=1, label=r'$\braket{q|L_2}$')
    ax2.plot(x, -1/2*R_2(x, TRAFMAT_SHO_ENERGY) + 1/2*(E_N[2]+E_N[3]) + E_b, "k-", linewidth=1, label=r'$\braket{q|R_2}$')
    ax2.hlines(y= E_N[0] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
    ax2.hlines(y= E_N[1] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
    ax2.hlines(y= E_N[2] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
    ax2.hlines(y= E_N[3] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
    ax2.text(6.5, E_N[0] +E_b -0.05, r'$\mathcal{E}_1$')
    ax2.text(6.5, E_N[1] +E_b -0.05, r'$\mathcal{E}_2$')
    ax2.text(6.5, E_N[2] +E_b -0.05, r'$\mathcal{E}_3$')
    ax2.text(6.5, E_N[3] +E_b -0.05, r'$\mathcal{E}_4$')
    ax2.text(3, -0.7, r'$E_{\text{b}}=$'+str(barrier))
    ax2.text(3, -0.8, r'$\epsilon=$'+str(bias))
    ax2.set_aspect(5)
    ax2.legend(frameon=False)

    plt.savefig('DDS_LOCALIZED_BASIS.png', dpi=300, bbox_inches='tight')

    fig, ax3 = plt.subplots()

    ax3.set_xlabel(r"$q$")
    ax3.set_ylabel(r"$V_0(q)$")
    ax3.set_xlim(-6, 6)
    ax3.set_ylim(-1, 2)
    ax3.set_xticks([-6,-4,-2,0,2,4,6])
    ax3.set_yticks([-0.8,-0.4,0,0.4,0.8,1.2,1.6,2])
    ax3.plot(x, 1/(64*E_b) *x**4 - 1/4 *x**2 +E_b - epsilon*x, color='k')
    ax3.plot(x, Q_1(x, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR), "k--", linewidth=1, label=r'$\braket{q|\alpha_1}$')
    ax3.plot(x, -Q_2(x, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR), "k:", linewidth=1, label=r'$\braket{q|\alpha_2}$')
    ax3.plot(x, -Q_3(x, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR), "k-.", linewidth=1, label=r'$\braket{q|\beta_2}$')
    ax3.plot(x, -Q_4(x, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR), "k-", linewidth=1, label=r'$\braket{q|\beta_1}$')
    ax3.plot(Q_DVR_BASIS[0,0], -1, 'x', zorder=10, clip_on=False, color='k')
    ax3.plot(Q_DVR_BASIS[1,1], -1, 'x', zorder=10, clip_on=False, color='k')
    ax3.plot(Q_DVR_BASIS[2,2], -1, 'x', zorder=10, clip_on=False, color='k')
    ax3.plot(Q_DVR_BASIS[3,3], -1, 'x', zorder=10, clip_on=False, color='k')
    ax3.text(Q_DVR_BASIS[0,0] -0.25, -1.25, r'$q_{\alpha_1}$')
    ax3.text(Q_DVR_BASIS[1,1] -0.25, -1.25, r'$q_{\alpha_2}$')
    ax3.text(Q_DVR_BASIS[2,2] -0.25, -1.25, r'$q_{\beta_2}$')
    ax3.text(Q_DVR_BASIS[3,3] -0.25, -1.25, r'$q_{\beta_1}$')
    ax3.text(3, -0.7, r'$E_{\text{b}}=$'+str(barrier))
    ax3.text(3, -0.8, r'$\epsilon=$'+str(bias))
    ax3.set_aspect(5)
    ax3.legend(frameon=False)

    plt.savefig('DDS_DVR_BASIS.png', dpi=300, bbox_inches='tight')

    return
