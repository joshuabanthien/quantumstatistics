import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

params = {'text.usetex': True,
          'text.latex.preamble': r'\usepackage{cmbright} \usepackage{amsmath}'}
plt.rcParams.update(params)

#set potential parameters and amount of SHO states considered

E_b = 1.4

epsilon = 0.05
#epsilon = 0

#E_b = float(input('Barrier height: '))

#epsilon = float(input('Bias: '))

K=48


#definition of kronecker delta and SHO states

def kronDelta(m,n):
    if m == n:
        return 1
    else:
        return 0

def SHO_n(n,x):
    return 1/np.sqrt(np.sqrt(np.pi) * 2**n *math.factorial(n)) * np.exp(-0.5*x**2) * scipy.special.hermite(n, monic=False)(x)


#definition of Hamiltonian matrix elements in SHO basis and consequent diagonalization

def m_H_n(m,n):
    return 1/4 * ( (2*n+1) * kronDelta(m,n) - np.sqrt(n*(n-1)) * kronDelta(m,n-2) - np.sqrt((n+1)*(n+2)) * kronDelta(m,n+2) ) \
        + 1/(256*E_b) * ( np.sqrt(n*(n-1)*(n-2)*(n-3)) * kronDelta(m,n-4) + 2*(2*n-1)*np.sqrt(n*(n-1))*kronDelta(m,n-2) + 3*(2*n**2+2*n+1)*kronDelta(m,n) + 2*(2*n+3)*np.sqrt((n+1)*(n+2))*kronDelta(m,n+2) + np.sqrt((n+1)*(n+2)*(n+3)*(n+4))*kronDelta(m,n+4) ) \
        - 1/8 * ( (2*n+1) * kronDelta(m,n) + np.sqrt(n*(n-1)) * kronDelta(m,n-2) + np.sqrt((n+1)*(n+2)) * kronDelta(m,n+2) ) \
        - epsilon/np.sqrt(2) * (np.sqrt(n) * kronDelta(m,n-1) + np.sqrt(n+1) * kronDelta(m,n+1))

H =  np.empty((K,K))

for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        H[i, j] = m_H_n(i,j)

E_n, energyEigenvectors_T = np.linalg.eigh(H)

energyEigenvectors = np.transpose(energyEigenvectors_T)


#calculation of transition amplitudes <m|q|n> and of matrix Q in localized state basis, consequent diagonalization

def m_Q_n(m,n):
    total = 0
    for i in range(0, K-1):
        total += energyEigenvectors[m-1][i] * energyEigenvectors[n-1][i+1] * np.sqrt(i+1)
    for i in range(0, K-1):
        total += energyEigenvectors[m-1][i] * energyEigenvectors[n-1][i-1] * np.sqrt(i)
    return 1/np.sqrt(2)*total

Q = 1/2 * np.array([[m_Q_n(1,1)+m_Q_n(2,2)-2*m_Q_n(1,2), m_Q_n(1,1)-m_Q_n(2,2), m_Q_n(1,3)-m_Q_n(1,4)-m_Q_n(2,3)+m_Q_n(2,4), m_Q_n(1,3)+m_Q_n(1,4)-m_Q_n(2,3)-m_Q_n(2,4)],
              [m_Q_n(1,1)-m_Q_n(2,2), m_Q_n(1,1)+m_Q_n(2,2)+2*m_Q_n(1,2), m_Q_n(1,3)-m_Q_n(1,4)+m_Q_n(2,3)-m_Q_n(2,4), m_Q_n(1,3)+m_Q_n(1,4)+m_Q_n(2,3)+m_Q_n(2,4)],
              [m_Q_n(1,3)-m_Q_n(1,4)-m_Q_n(2,3)+m_Q_n(2,4), m_Q_n(1,3)-m_Q_n(1,4)+m_Q_n(2,3)-m_Q_n(2,4), m_Q_n(3,3)+m_Q_n(4,4)-2*m_Q_n(3,4), m_Q_n(3,3)-m_Q_n(4,4)],
              [m_Q_n(1,3)+m_Q_n(1,4)-m_Q_n(2,3)-m_Q_n(2,4), m_Q_n(1,3)+m_Q_n(1,4)+m_Q_n(2,3)+m_Q_n(2,4), m_Q_n(3,3)-m_Q_n(4,4), m_Q_n(3,3)+m_Q_n(4,4)+2*m_Q_n(3,4)]])

q_n, positionEigenvectors_T = np.linalg.eigh(Q)

positionEigenvectors = np.transpose(positionEigenvectors_T)

inv_positionEigenvectors = np.linalg.inv(positionEigenvectors)

H_loc = 1/2 * np.array([[E_n[0]+E_n[1], E_n[0]-E_n[1], 0, 0],
                        [E_n[0]-E_n[1], E_n[0]+E_n[1], 0, 0],
                        [0, 0, E_n[2]+E_n[3], E_n[2]-E_n[3]],
                        [0, 0, E_n[2]-E_n[3], E_n[2]+E_n[3]]])

H_DVR = positionEigenvectors.dot(H_loc).dot(inv_positionEigenvectors)

E_mu, DVR_energy_eigenvector = np.linalg.eigh(H_DVR)

print(E_mu)
print(H_DVR)


#printing

print(f'First four energy levels: E_1 = {E_n[0]:.3f}, E_2 = {E_n[1]:.3f}, E_3 = {E_n[2]:.3f}, E_4 = {E_n[3]:.3f}')
print(f'Position eigenvalues: q_alpha_1 = {q_n[0]:.3f}, q_alpha_2 = {q_n[1]:.3f}, q_beta_2 = {q_n[2]:.3f}, q_beta_1 = {q_n[3]:.3f}')


#plotting

def Psi_n(n,x):
    result = 0
    for i in range(0, K-1):
        result += energyEigenvectors[n-1][i] * SHO_n(i,x)
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
    return positionEigenvectors[0][0]*L_1(x) + positionEigenvectors[0][1]*R_1(x) + positionEigenvectors[0][2]*L_2(x) +positionEigenvectors[0][3]*R_2(x)

def q_2(x):
    return positionEigenvectors[1][0]*L_1(x) + positionEigenvectors[1][1]*R_1(x) + positionEigenvectors[1][2]*L_2(x) +positionEigenvectors[1][3]*R_2(x)

def q_3(x):
    return positionEigenvectors[2][0]*L_1(x) + positionEigenvectors[2][1]*R_1(x) + positionEigenvectors[2][2]*L_2(x) +positionEigenvectors[2][3]*R_2(x)

def q_4(x):
    return positionEigenvectors[3][0]*L_1(x) + positionEigenvectors[3][1]*R_1(x) + positionEigenvectors[3][2]*L_2(x) +positionEigenvectors[3][3]*R_2(x)


x=np.linspace(-6,6,600)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r"$q$")
ax1.set_ylabel(r"$\mathsf{V_0(q)}$")
ax1.set_xlim(-6, 6)
ax1.set_ylim(-1, 2)
ax1.set_xticks([-6,-4,-2,0,2,4,6])
ax1.set_yticks([-0.8,-0.4,0,0.4,0.8,1.2,1.6,2])
ax1.plot(x, 1/(64*E_b) *x**4 - 1/4 *x**2 +E_b - epsilon*x, color='k')
ax1.plot(x, -1/2*Psi_n(1,x) + E_n[0] + E_b, "k--", linewidth=1, label=r'$\psi_1$')
ax1.plot(x, 1/2*Psi_n(2,x) + E_n[1] + E_b, "k:", linewidth=1, label=r'$\psi_2$')
ax1.plot(x, -1/2*Psi_n(3,x) + E_n[2] + E_b, "k-.", linewidth=1, label=r'$\psi_3$')
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
ax2.plot(x, -1/2*L_1(x) + 1/2*(E_n[0]+E_n[1]) + E_b, "k--", linewidth=1, label=r'$L_1$')
ax2.plot(x, -1/2*R_1(x) + 1/2*(E_n[0]+E_n[1]) + E_b, "k:", linewidth=1, label=r'$R_1$')
ax2.plot(x, -1/2*L_2(x) + 1/2*(E_n[2]+E_n[3]) + E_b, "k-.", linewidth=1, label=r'$L_2$')
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
ax3.plot(x, q_1(x), "k--", linewidth=1, label=r'$\langle q | \alpha_1 \rangle$')
ax3.plot(x, -q_2(x), "k:", linewidth=1, label=r'$\langle q | \alpha_2 \rangle$')
ax3.plot(x, q_3(x), "k-.", linewidth=1, label=r'$\langle q | \beta_2 \rangle$')
ax3.plot(x, q_4(x), "k-", linewidth=1, label=r'$\langle q | \beta_1 \rangle$')
ax3.plot(q_n[0], -1, 'x', zorder=10, clip_on=False, color='k')
ax3.plot(q_n[1], -1, 'x', zorder=10, clip_on=False, color='k')
ax3.plot(q_n[2], -1, 'x', zorder=10, clip_on=False, color='k')
ax3.plot(q_n[3], -1, 'x', zorder=10, clip_on=False, color='k')
ax3.text(q_n[0] -0.25, -1.25, r'$q_{\alpha_1}$')
ax3.text(q_n[1] -0.25, -1.25, r'$q_{\alpha_2}$')
ax3.text(q_n[2] -0.25, -1.25, r'$q_{\beta_2}$')
ax3.text(q_n[3] -0.25, -1.25, r'$q_{\beta_1}$')
ax3.set_aspect(5)
ax3.legend(frameon=False)

plt.savefig('test3.png', dpi=300, bbox_inches='tight')

plt.show()
