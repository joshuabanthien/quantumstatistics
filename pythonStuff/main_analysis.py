import numpy as np
import dds
import dds_plots
import rate_matrix
import sim_density_matrix
import sim_density_matrix_plots
import matplotlib.pyplot as plt
import linear_response_function
import linear_response_plots


params = {'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath} \usepackage{braket}'}
plt.rcParams.update(params)


def MAIN_ANALYSIS(n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq):

    H_DVR, Q_DVR, E_N, TRAFMAT_SHO_ENERGY, TRAFMAT_ENERGY_LOC, TRAFMAT_LOC_DVR = dds.H_DVR_Q_DVR(n_sys, barrier, bias)

    dds_plots.GENERATE_DDS_PLOTS(Q_DVR, E_N, TRAFMAT_SHO_ENERGY, TRAFMAT_LOC_DVR, barrier, bias)

    RATE_MATRIX_APPROX = rate_matrix.RATE_MATRIX(H_DVR, Q_DVR, omega, gamma, coupling, temp)

    RATE_MATRIX_UNAPPROX = rate_matrix.RATE_MATRIX_UNAPPROX(H_DVR, Q_DVR, omega, gamma, coupling, temp)

    RATE_MATRIX_DRIVEN_UNAPPROX = rate_matrix.RATE_MATRIX_DRIVEN_UNAPPROX(H_DVR, Q_DVR, omega, gamma, coupling, temp, drive_coup, drive_freq)

    #RATE_MATRIX = rate_matrix.RATE_MATRIX_UNAPPROX(H_DVR, Q_DVR, omega, gamma, coupling, temp)

    #RATE_MATRIX = rate_matrix.RATE_MATRIX_DRIVEN_UNAPPROX(H_DVR, Q_DVR, omega, gamma, coupling, temp, 5, 2)

    RHO_INIT = np.array([1,0,0,0])

    SIM_TIME = np.linspace(0, 1000, 5000)

    RHO_EQUI = sim_density_matrix.DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_APPROX, SIM_TIME)[4999,:]

    RHO_S = sim_density_matrix.DIAG_DENSITY_MATRIX_ELEMENTS(RHO_EQUI, RATE_MATRIX_DRIVEN_UNAPPROX, SIM_TIME)

    LINEAR_RESPONSE = linear_response_function.LINEAR_RESPONSE_FUNCTION(RHO_EQUI, RHO_S, Q_DVR, SIM_TIME)

    linear_response_plots.GENERATE_LINEAR_RESPONSE_PLOTS(SIM_TIME,LINEAR_RESPONSE)

    DIAG_DENSITY_MATRIX_ELEMENTS_APPROX = sim_density_matrix.DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_APPROX, SIM_TIME)

    DIAG_DENSITY_MATRIX_ELEMENTS_UNAPPROX = sim_density_matrix.DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_UNAPPROX, SIM_TIME)

    DIAG_DENSITY_MATRIX_ELEMENTS_DRIVEN_APPROX = sim_density_matrix.DIAG_DENSITY_MATRIX_ELEMENTS(RHO_INIT, RATE_MATRIX_DRIVEN_UNAPPROX, SIM_TIME)

    sim_density_matrix_plots.GENERATE_DENSITY_MATRIX_PLOTS_APPROX(DIAG_DENSITY_MATRIX_ELEMENTS_APPROX, SIM_TIME, n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq)

    sim_density_matrix_plots.GENERATE_DENSITY_MATRIX_PLOTS_UNAPPROX(DIAG_DENSITY_MATRIX_ELEMENTS_UNAPPROX, SIM_TIME, n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq)

    sim_density_matrix_plots.GENERATE_DENSITY_MATRIX_PLOTS_DRIVEN_UNAPPROX(DIAG_DENSITY_MATRIX_ELEMENTS_DRIVEN_APPROX, SIM_TIME, n_sys, barrier, bias, omega, gamma, coupling, temp, drive_coup, drive_freq)

    return

MAIN_ANALYSIS(1, 1.4, 0.05, 1, 0.097, 0.18, 1, 0.4, 2)
