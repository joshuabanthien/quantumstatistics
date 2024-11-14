import scipy
import rate_matrix
import dds
import numpy as np
import matplotlib.pyplot as plt
import lmfit


params = {'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath} \usepackage{braket}'}
plt.rcParams.update(params)


T = np.linspace(0,10000,10000)
Y0 = np.array([1,0,0,0])

n_sys = 1
barrier = 1.4
bias = 0.05
omega = 1
gamma = 0.097
coupling = 0.18
temp = 1


def integrate_diff_eq(Y_INIT, TIME, n_sys, barrier, bias, omega, gamma, coupling, temp):

        H_DVR, Q_DVR = dds.H_DVR_Q_DVR(n_sys, barrier, bias)[0], dds.H_DVR_Q_DVR(n_sys, barrier, bias)[1]

        M = rate_matrix.RATE_MATRIX(H_DVR, Q_DVR, omega, gamma, coupling, temp)

        def func(y ,t):

            return M @ y

        Y = scipy.integrate.odeint(func, Y_INIT, TIME)

        return Y


Y = integrate_diff_eq(Y0, T, n_sys, barrier, bias, omega, gamma, coupling, temp)
#logY = np.log(Y)



def double_exp_decay(x, a1, c1, a2, c2, b, d):

        return a1*np.exp(-x*c1) + a2*np.exp(-(x-d)*c2) + b


def fit_data(Y, T):

    Y_DAT = Y[:,0]

    P0 = np.array([5.0, 0.25, 5.0, 0.01, 5.0, 5.0])

    params = scipy.optimize.curve_fit(double_exp_decay, T, Y_DAT, P0, bounds=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[10.0, 0.7, 10.0, 0.02, 10.0, 10.0]], maxfev=2000)

    return params[0]


#mymodel = lmfit.Model(double_exp_decay)
#params = mymodel.make_params(a1=dict(value=1.0, min=0, max=10), c1=dict(value=1.0, min=0, max=10), a2=dict(value=1.0, min=0, max=10), c2=dict(value=1.0, min=0, max=10), b=dict(value=1.0, min=0, max=10), d=dict(value=1.0, min=0, max=100))
#result = mymodel.fit(Y[:,0], params, x=T)

#plt.plot(T, result.best_fit)
#plt.show()

#test_params = fit_data(Y, T)

#print(test_params)

#Y_FIT = double_exp_decay(T, test_params[0], test_params[1], test_params[2], test_params[3], test_params[4], test_params[5])

#plt.plot(T, logY[:,0])
#plt.plot(T, Y_FIT)
#plt.show()


def intra_decay_rate(Y_INIT, TIME, n_sys, barrier, bias, omega, gamma, coupling, temp):

    Y = integrate_diff_eq(Y_INIT, TIME, n_sys, barrier, bias, omega, gamma, coupling, temp)

    FIT_PARAMS = fit_data(Y, TIME)

    return max(FIT_PARAMS[1], FIT_PARAMS[3])


def inter_decay_rate(Y_INIT, TIME, n_sys, barrier, bias, omega, gamma, coupling, temp):

    Y = integrate_diff_eq(Y_INIT, TIME, n_sys, barrier, bias, omega, gamma, coupling, temp)

    FIT_PARAMS = fit_data(Y, TIME)

    return min(FIT_PARAMS[1], FIT_PARAMS[3])


def print_fit_params(Y_INIT, TIME, n_sys, barrier, bias, omega, gamma, coupling, temp):

    Y = integrate_diff_eq(Y_INIT, TIME, n_sys, barrier, bias, omega, gamma, coupling, temp)

    FIT_PARAMS = fit_data(Y, TIME)

    return FIT_PARAMS[1], FIT_PARAMS[3]

#TEMP = np.linspace(0.3,1,20)
OMEGA = np.linspace(0.3,5,100)
#GAMMA = np.linspace(0.01, 1.0, 100)
#COUPLING = np.linspace(0.01, 0.4, 100)

INTER_DECAY_RATE_OMEGA = np.empty(len(OMEGA))
#INTER_DECAY_RATE_GAMMA = np.empty(len(GAMMA))
#INTER_DECAY_RATE_COUPLING = np.empty(len(COUPLING))

#for i in range(len(OMEGA)):

#    print(print_fit_params(Y0, T, n_sys, barrier, bias, OMEGA[i], 0.5, coupling, temp))

#for i in range(len(GAMMA)):

#    print(print_fit_params(Y0, T, n_sys, barrier, bias, 0.5, GAMMA[i], coupling, temp))

#for i in range(len(COUPLING)):

#    print(print_fit_params(Y0, T, n_sys, barrier, bias, omega, 0.01, COUPLING[i], temp))




#plt.plot(OMEGA, INTER_DECAY_RATE_OMEGA)
#plt.show()


def analysis():

    n_sys = 1
    barrier = 1.4
    bias = 0.05
    omega = 1
    OMEGA = [0.6, 0.8, 1, 1.5]
    GAMMA = [0.01, 0.05, 0.097, 0.5]
    gamma = 0.097
    coupling = 0.18
    COUPLING = [0.18, 0.3]
    TEMP = [0.3, 0.5, 0.7, 1, 2, 5, 10]
    Y0 = np.array([1,0,0,0])
    T = np.linspace(0,10000,10000)


    COLORS = ['red', 'blue', 'green', 'black', 'violet', 'yellow', 'orange', 'turquoise']


    fig1, ax1 = plt.subplots()

    X_OMEGA = np.linspace(0.3,5,100)

    #for gamma, color in zip(GAMMA, COLORS):

    #    INTRA_DECAY_RATE_OMEGA = np.empty(len(X_OMEGA))

    #    for i in range(len(X_OMEGA)):

    #        INTRA_DECAY_RATE_OMEGA[i] = intra_decay_rate(Y0, T, n_sys, barrier, bias, X_OMEGA[i], gamma, coupling, TEMP[3])

    #    ax1.plot(X_OMEGA,INTRA_DECAY_RATE_OMEGA, color=color,  label=r"$\Gamma={}$".format(gamma))


    ax1.set_xlim(0.3,5)
    ax1.set_ylim(0,0.7)
    ax1.set_xlabel(r'$\Omega$')
    ax1.set_ylabel(r'$\Gamma_{\text{intra}}$')
    ax1.text(2.0,0.65, r'$E_\text{b}=$'+str(barrier))
    ax1.text(2.75,0.65, r'$\epsilon=$'+str(bias))
    ax1.text(2.0,0.60, r'$g=$'+str(coupling))
    ax1.text(2.75,0.60, r'$T=$'+str(temp))
    ax1.legend(frameon=False)
    fig1.savefig('INTRA_RATE_GAMMA_OMEGA.png', dpi=300, bbox_inches='tight')

    print('Fig1 finished!')


    fig2, ax2 = plt.subplots()

    X_GAMMA = np.linspace(0.001,1.0,100)

    #for omega, color in zip(OMEGA, COLORS):

    #    INTRA_DECAY_RATE_GAMMA = np.empty(len(X_GAMMA))

    #    for i in range(len(X_GAMMA)):

    #        INTRA_DECAY_RATE_GAMMA[i] = intra_decay_rate(Y0, T, n_sys, barrier, bias, omega, X_GAMMA[i], coupling, TEMP[3])

    #    ax2.plot(X_GAMMA,INTRA_DECAY_RATE_GAMMA, color=color,  label=r"$\Omega={}$".format(omega))


    ax2.set_xlim(0.01,1)
    ax2.set_ylim(0,0.7)
    ax2.set_xlabel(r'$\Gamma$')
    ax2.set_ylabel(r'$\Gamma_{\text{intra}}$')
    ax2.text(0.4,0.65, r'$E_\text{b}=$'+str(barrier))
    ax2.text(0.55,0.65, r'$\epsilon=$'+str(bias))
    ax2.text(0.4,0.60, r'$g=$'+str(coupling))
    ax2.text(0.55,0.60, r'$T=$'+str(temp))
    ax2.legend(frameon=False)
    fig2.savefig('INTRA_RATE_OMEGA_GAMMA.png', dpi=300, bbox_inches='tight')

    print('Fig2 finished!')


    fig3, ax3 = plt.subplots()

    X_OMEGA = np.linspace(0.3,5,100)

    #for gamma, color in zip(GAMMA, COLORS):

    #    INTER_DECAY_RATE_OMEGA = np.empty(len(X_OMEGA))

    #    for i in range(len(X_OMEGA)):

    #        INTER_DECAY_RATE_OMEGA[i] = inter_decay_rate(Y0, T, n_sys, barrier, bias, X_OMEGA[i], gamma, coupling, TEMP[3])

    #    ax3.plot(X_OMEGA,INTER_DECAY_RATE_OMEGA, color=color,  label=r"$\Gamma={}$".format(gamma))


    ax3.set_xlim(0.3,5)
    ax3.set_ylim(0,0.012)
    ax3.set_xlabel(r'$\Omega$')
    ax3.set_ylabel(r'$\Gamma_{\text{inter}}$')
    ax3.text(2,0.011, r'$E_\text{b}=$'+str(barrier))
    ax3.text(2.75,0.011, r'$\epsilon=$'+str(bias))
    ax3.text(2,0.010, r'$g=$'+str(coupling))
    ax3.text(2.75,0.010, r'$T=$'+str(temp))
    ax3.legend(frameon=False)
    fig3.savefig('INTER_RATE_GAMMA_OMEGA.png', dpi=300, bbox_inches='tight')

    print('Fig3 finished!')


    fig4, ax4 = plt.subplots()

    X_GAMMA = np.linspace(0.01,1,100)

    #for omega, color in zip(OMEGA, COLORS):

    #    INTER_DECAY_RATE_GAMMA = np.empty(len(X_GAMMA))

    #    for i in range(len(X_GAMMA)):

    #        INTER_DECAY_RATE_GAMMA[i] = inter_decay_rate(Y0, T, n_sys, barrier, bias, omega, X_GAMMA[i], coupling, TEMP[3])

    #    ax4.plot(X_GAMMA,INTER_DECAY_RATE_GAMMA, color=color,  label=r"$\Omega={}$".format(omega))


    ax4.set_xlim(0.01,1)
    ax4.set_ylim(0,0.006)
    ax4.set_xlabel(r'$\Gamma$')
    ax4.set_ylabel(r'$\Gamma_{\text{inter}}$')
    ax4.text(0.5,0.0055, r'$E_\text{b}=$'+str(barrier))
    ax4.text(0.65,0.0055, r'$\epsilon=$'+str(bias))
    ax4.text(0.5,0.005, r'$g=$'+str(coupling))
    ax4.text(0.65,0.005, r'$T=$'+str(temp))
    ax4.legend(frameon=False)
    fig4.savefig('INTER_RATE_OMEGA_GAMMA.png', dpi=300, bbox_inches='tight')

    print('Fig4 finished!')


    fig5, ax5 = plt.subplots()

    X_COUPLING = np.linspace(0.01,0.4,100)

    #for omega, color in zip(OMEGA, COLORS):

    #    INTER_DECAY_RATE_COUPLING = np.empty(len(X_COUPLING))

    #    for i in range(len(X_OMEGA)):

    #        INTER_DECAY_RATE_COUPLING[i] = inter_decay_rate(Y0, T, n_sys, barrier, bias, omega, gamma, X_COUPLING[i], TEMP[3])

    #    ax5.plot(X_COUPLING,INTER_DECAY_RATE_COUPLING, color=color,  label=r"$\Omega={}$".format(omega))


    ax5.set_xlim(0.01,0.4)
    ax5.set_ylim(0,0.009)
    ax5.set_xlabel(r'$g$')
    ax5.set_ylabel(r'$\Gamma_{\text{inter}}$')
    ax5.text(0.15,0.008, r'$E_\text{b}=$'+str(barrier))
    ax5.text(0.2,0.008, r'$\epsilon=$'+str(bias))
    ax5.text(0.15,0.0075, r'$\Gamma=$'+str(gamma))
    ax5.text(0.2,0.0075, r'$T=$'+str(temp))
    ax5.legend(frameon=False)
    fig5.savefig('INTER_RATE_OMEGA_COUPLING.png', dpi=300, bbox_inches='tight')

    print('Fig5 finished!')


    fig6, ax6 = plt.subplots()

    X_COUPLING = np.linspace(0.01,0.4,100)

    for gamma, color in zip(GAMMA, COLORS):

        INTER_DECAY_RATE_COUPLING = np.empty(len(X_COUPLING))

        for i in range(len(X_COUPLING)):

            INTER_DECAY_RATE_COUPLING[i] = inter_decay_rate(Y0, T, n_sys, barrier, bias, omega, gamma, X_COUPLING[i], TEMP[3])

        ax6.plot(X_COUPLING,INTER_DECAY_RATE_COUPLING, color=color,  label=r"$\Gamma={}$".format(gamma))


    ax6.set_xlim(0.01,0.4)
    ax6.set_ylim(0,0.012)
    ax6.set_xlabel(r'$g$')
    ax6.set_ylabel(r'$\Gamma_{\text{inter}}$')
    ax6.text(0.15,0.011, r'$E_\text{b}=$'+str(barrier))
    ax6.text(0.20,0.011, r'$\epsilon=$'+str(bias))
    ax6.text(0.15,0.010, r'$\Omega=$'+str(omega))
    ax6.text(0.20,0.010, r'$T=$'+str(temp))
    ax6.legend(frameon=False)
    fig6.savefig('INTER_RATE_GAMMA_COUPLING.png', dpi=300, bbox_inches='tight')

    print('Fig6 finished!')


analysis()
