import numpy as np
import rate_matrix
import dds
import matplotlib.pyplot as plt
import sim_density_matrix


params = {'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath} \usepackage{braket}'}
plt.rcParams.update(params)


def GENERATE_DENSITY_MATRIX_PLOTS_APPROX(Y_DATA, T_DATA, n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq):

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"$t$ in $\omega_0^{-1}$")
    ax1.set_xlim(0, max(T_DATA))
    ax1.set_ylim(0, 1.1)
    ax1.plot(T_DATA,Y_DATA[:,0], 'k-', linewidth=1, label=r"$\rho_{11}$")
    ax1.plot(T_DATA,Y_DATA[:,1], 'k--', linewidth=1, label=r"$\rho_{22}$")
    ax1.plot(T_DATA,Y_DATA[:,2], 'k-.', linewidth=1, label=r"$\rho_{33}$")
    ax1.plot(T_DATA,Y_DATA[:,3], 'k:', linewidth=1, label=r"$\rho_{44}$")
    ax1.text(500, 1, r'$E_{\text{b}}=$'+str(barrier))
    ax1.text(500, 0.95, r'$\epsilon=$'+str(bias))
    ax1.text(500, 0.9, r'$n_{\text{sys}}=$'+str(n_sys))
    ax1.text(500, 0.85, r'$\Omega=$'+str(omega))
    ax1.text(650, 1, r'$\Gamma=$'+str(gamma))
    ax1.text(650, 0.95, r'$g=$'+str(coupling))
    ax1.text(650, 0.9, r'$T=$'+str(temp))
    ax1.legend(frameon=False)

    plt.show()

    return


def GENERATE_DENSITY_MATRIX_PLOTS_UNAPPROX(Y_DATA, T_DATA, n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq):

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"$t$ in $\omega_0^{-1}$")
    ax1.set_xlim(0, max(T_DATA))
    ax1.set_ylim(0, 1.1)
    ax1.plot(T_DATA,Y_DATA[:,0], 'k-', linewidth=1, label=r"$\rho_{11}$")
    ax1.plot(T_DATA,Y_DATA[:,1], 'k--', linewidth=1, label=r"$\rho_{22}$")
    ax1.plot(T_DATA,Y_DATA[:,2], 'k-.', linewidth=1, label=r"$\rho_{33}$")
    ax1.plot(T_DATA,Y_DATA[:,3], 'k:', linewidth=1, label=r"$\rho_{44}$")
    ax1.text(500, 1, r'$E_{\text{b}}=$'+str(barrier))
    ax1.text(500, 0.95, r'$\epsilon=$'+str(bias))
    ax1.text(500, 0.9, r'$n_{\text{sys}}=$'+str(n_sys))
    ax1.text(500, 0.85, r'$\Omega=$'+str(omega))
    ax1.text(650, 1, r'$\Gamma=$'+str(gamma))
    ax1.text(650, 0.95, r'$g=$'+str(coupling))
    ax1.text(650, 0.9, r'$T=$'+str(temp))
    ax1.legend(frameon=False)

    plt.show()

    return


def GENERATE_DENSITY_MATRIX_PLOTS_DRIVEN_UNAPPROX(Y_DATA, T_DATA, n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq):

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"$t$ in $\omega_0^{-1}$")
    ax1.set_xlim(0, max(T_DATA))
    ax1.set_ylim(0, 1.1)
    ax1.plot(T_DATA,Y_DATA[:,0], 'k-', linewidth=1, label=r"$\rho_{11}$")
    ax1.plot(T_DATA,Y_DATA[:,1], 'k--', linewidth=1, label=r"$\rho_{22}$")
    ax1.plot(T_DATA,Y_DATA[:,2], 'k-.', linewidth=1, label=r"$\rho_{33}$")
    ax1.plot(T_DATA,Y_DATA[:,3], 'k:', linewidth=1, label=r"$\rho_{44}$")
    ax1.text(500, 1, r'$E_{\text{b}}=$'+str(barrier))
    ax1.text(500, 0.95, r'$\epsilon=$'+str(bias))
    ax1.text(500, 0.9, r'$n_{\text{sys}}=$'+str(n_sys))
    ax1.text(500, 0.85, r'$\Omega=$'+str(omega))
    ax1.text(500, 0.8, r'$s=$'+str(drive_coup))
    ax1.text(650, 1, r'$\Gamma=$'+str(gamma))
    ax1.text(650, 0.95, r'$g=$'+str(coupling))
    ax1.text(650, 0.9, r'$T=$'+str(temp))
    ax1.text(650, 0.85, r'$\Omega_{\text{d}}=$'+str(drive_freq))
    ax1.legend(frameon=False)

    plt.show()

    return
