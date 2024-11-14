import numpy as np
import rate_matrix
import dds
import matplotlib.pyplot as plt
import sim_density_matrix


params = {'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath} \usepackage{braket}'}
plt.rcParams.update(params)


def GENERATE_LINEAR_RESPONSE_PLOTS(T_DATA,Y_DATA):

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"$t$ in $\omega_0^{-1}$")
    ax1.set_xlim(0, max(T_DATA))
    #ax1.set_ylim(0, 1.1)
    ax1.plot(T_DATA,Y_DATA, 'k-', linewidth=1)
    #ax1.text(500, 1, r'$E_{\text{b}}=$'+str(barrier))
    #ax1.text(500, 0.95, r'$\epsilon=$'+str(bias))
    #ax1.text(500, 0.9, r'$n_{\text{sys}}=$'+str(n_sys))
    #ax1.text(500, 0.85, r'$\Omega=$'+str(omega))
    #ax1.text(650, 1, r'$\Gamma=$'+str(gamma))
    #ax1.text(650, 0.95, r'$g=$'+str(coupling))
    #ax1.text(650, 0.9, r'$T=$'+str(temp))
    #ax1.legend(frameon=False)

    plt.show()

    return
