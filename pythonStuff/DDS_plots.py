import matplotlib.pyplot as plt
import numpy as np
import DDS as dds
import helpful_functions as hf


params = {'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath}'}
plt.rcParams.update(params)


K=50
E_b = 1.4
epsilon = 0.05
n_sys=1


H_DVR_BASIS, Q_DVR_BASIS, E_n, trafmat_SHO_ENERGY, trafmat_ENERGY_LOC, trafmat_LOC_DVR = dds.calc_H_DVR_Q_DVR(n_sys,E_b,epsilon)


def Psi_n(n,x):
    result = 0
    for i in range(0, K-1):
        result += trafmat_SHO_ENERGY[n-1][i] * hf.SHO_n(i,x)
    return result


def L_1(x):
    return 1/np.sqrt(2) * (Psi_n(1,x)-Psi_n(2,x))

def R_1(x):
    return 1/np.sqrt(2) * (Psi_n(1,x)+Psi_n(2,x))

def L_2(x):
    return 1/np.sqrt(2) * (Psi_n(3,x)-Psi_n(4,x))

def R_2(x):
    return 1/np.sqrt(2) * (Psi_n(3,x)+Psi_n(4,x))


def q_1(x):
    return trafmat_LOC_DVR[0][0]*L_1(x) + trafmat_LOC_DVR[0][1]*R_1(x) + trafmat_LOC_DVR[0][2]*L_2(x) + trafmat_LOC_DVR[0][3]*R_2(x)

def q_2(x):
    return trafmat_LOC_DVR[1][0]*L_1(x) + trafmat_LOC_DVR[1][1]*R_1(x) + trafmat_LOC_DVR[1][2]*L_2(x) + trafmat_LOC_DVR[1][3]*R_2(x)

def q_3(x):
    return trafmat_LOC_DVR[2][0]*L_1(x) + trafmat_LOC_DVR[2][1]*R_1(x) + trafmat_LOC_DVR[2][2]*L_2(x) + trafmat_LOC_DVR[2][3]*R_2(x)

def q_4(x):
    return trafmat_LOC_DVR[3][0]*L_1(x) + trafmat_LOC_DVR[3][1]*R_1(x) + trafmat_LOC_DVR[3][2]*L_2(x) + trafmat_LOC_DVR[3][3]*R_2(x)


x=np.linspace(-6,6,600)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r"$q$")
ax1.set_ylabel(r"$\mathsf{V_0(q)}$")
ax1.set_xlim(-6, 6)
ax1.set_ylim(-1, 2)
ax1.set_xticks([-6,-4,-2,0,2,4,6])
ax1.set_yticks([-0.8,-0.4,0,0.4,0.8,1.2,1.6,2])
ax1.plot(x, 1/(64*E_b) *x**4 - 1/4 *x**2 +E_b - epsilon*x, color='k')
ax1.plot(x, 1/2*Psi_n(1,x) + E_n[0] + E_b, "k--", linewidth=1, label=r'$\psi_1$')
ax1.plot(x, -1/2*Psi_n(2,x) + E_n[1] + E_b, "k:", linewidth=1, label=r'$\psi_2$')
ax1.plot(x, 1/2*Psi_n(3,x) + E_n[2] + E_b, "k-.", linewidth=1, label=r'$\psi_3$')
ax1.plot(x, 1/2*Psi_n(4,x) + E_n[3] + E_b, "k-", linewidth=1, label=r'$\psi_4$')
ax1.hlines(y= E_n[0] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
ax1.hlines(y= E_n[1] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
ax1.hlines(y= E_n[2] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
ax1.hlines(y= E_n[3] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
ax1.text(6.5, E_n[0] +E_b -0.05, r'$\mathcal{E}_1$')
ax1.text(6.5, E_n[1] +E_b -0.05, r'$\mathcal{E}_2$')
ax1.text(6.5, E_n[2] +E_b -0.05, r'$\mathcal{E}_3$')
ax1.text(6.5, E_n[3] +E_b -0.05, r'$\mathcal{E}_4$')
ax1.set_aspect(5)
ax1.legend(frameon=False)

plt.savefig('test1.png', dpi=300, bbox_inches='tight')

fig, ax2 = plt.subplots()

ax2.set_xlabel(r"$q$")
ax2.set_ylabel(r"$V_0(q)$")
ax2.set_xlim(-6, 6)
ax2.set_ylim(-1, 2)
ax2.set_xticks([-6,-4,-2,0,2,4,6])
ax2.set_yticks([-0.8,-0.4,0,0.4,0.8,1.2,1.6,2])
ax2.plot(x, 1/(64*E_b) *x**4 - 1/4 *x**2 +E_b - epsilon*x, color='k')
ax2.plot(x, 1/2*L_1(x) + 1/2*(E_n[0]+E_n[1]) + E_b, "k--", linewidth=1, label=r'$L_1$')
ax2.plot(x, -1/2*R_1(x) + 1/2*(E_n[0]+E_n[1]) + E_b, "k:", linewidth=1, label=r'$R_1$')
ax2.plot(x, 1/2*L_2(x) + 1/2*(E_n[2]+E_n[3]) + E_b, "k-.", linewidth=1, label=r'$L_2$')
ax2.plot(x, -1/2*R_2(x) + 1/2*(E_n[2]+E_n[3]) + E_b, "k-", linewidth=1, label=r'$R_2$')
ax2.hlines(y= E_n[0] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
ax2.hlines(y= E_n[1] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
ax2.hlines(y= E_n[2] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
ax2.hlines(y= E_n[3] +E_b, xmin=-6, xmax=6, linewidth=0.5, color='k')
ax2.text(6.5, E_n[0] +E_b -0.05, r'$\mathcal{E}_1$')
ax2.text(6.5, E_n[1] +E_b -0.05, r'$\mathcal{E}_2$')
ax2.text(6.5, E_n[2] +E_b -0.05, r'$\mathcal{E}_3$')
ax2.text(6.5, E_n[3] +E_b -0.05, r'$\mathcal{E}_4$')
ax2.set_aspect(5)
ax2.legend(frameon=False)

plt.savefig('test2.png', dpi=300, bbox_inches='tight')

fig, ax3 = plt.subplots()

ax3.set_xlabel(r"$q$")
ax3.set_ylabel(r"$V_0(q)$")
ax3.set_xlim(-6, 6)
ax3.set_ylim(-1, 2)
ax3.set_xticks([-6,-4,-2,0,2,4,6])
ax3.set_yticks([-0.8,-0.4,0,0.4,0.8,1.2,1.6,2])
ax3.plot(x, 1/(64*E_b) *x**4 - 1/4 *x**2 +E_b - epsilon*x, color='k')
ax3.plot(x, -q_1(x), "k--", linewidth=1, label=r'$\langle q | \alpha_1 \rangle$')
ax3.plot(x, -q_2(x), "k:", linewidth=1, label=r'$\langle q | \alpha_2 \rangle$')
ax3.plot(x, -q_3(x), "k-.", linewidth=1, label=r'$\langle q | \beta_2 \rangle$')
ax3.plot(x, q_4(x), "k-", linewidth=1, label=r'$\langle q | \beta_1 \rangle$')
ax3.plot(Q_DVR_BASIS[0,0], -1, 'x', zorder=10, clip_on=False, color='k')
ax3.plot(Q_DVR_BASIS[1,1], -1, 'x', zorder=10, clip_on=False, color='k')
ax3.plot(Q_DVR_BASIS[2,2], -1, 'x', zorder=10, clip_on=False, color='k')
ax3.plot(Q_DVR_BASIS[3,3], -1, 'x', zorder=10, clip_on=False, color='k')
ax3.text(Q_DVR_BASIS[0,0] -0.25, -1.25, r'$q_{\alpha_1}$')
ax3.text(Q_DVR_BASIS[1,1] -0.25, -1.25, r'$q_{\alpha_2}$')
ax3.text(Q_DVR_BASIS[2,2] -0.25, -1.25, r'$q_{\beta_2}$')
ax3.text(Q_DVR_BASIS[3,3] -0.25, -1.25, r'$q_{\beta_1}$')
ax3.set_aspect(5)
ax3.legend(frameon=False)

plt.savefig('test3.png', dpi=300, bbox_inches='tight')

plt.show()
