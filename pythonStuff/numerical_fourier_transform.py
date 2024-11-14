import numpy as np
import cmath


def fourier_transform_real(T,x_in,y_in):

    y_in_complex = y_in

    y_out_complex = np.empty(len(y_in_complex), dtype=complex)

    y_out_complex = np.empty(len(y_in_real), dtype=complex)

    for i in range(len(x_in)):

        y_out_complex[i] = y_in_complex[i]*np.exp(1j*t*x_in[i])

    y_real_integrand = np.real(y_out_complex)

    val = np.trapz(y_real_integrand,x_in)

    return val


def fourier_transform_imag(t,x_in,y_in):

    y_in_complex = y_in

    y_out_complex = np.empty(len(y_in_complex), dtype=complex)

    for i in range(len(x_in)):

        y_out_complex[i] = y_in_complex[i]*np.exp(1j*t*x_in[i])

    y_imag_integrand = np.imag(y_out_complex)

    val = np.trapz(y_imag_integrand,x_in)

    return val


def fourier_transform_imag_integrand(t,x_in,y_in):

    y_in_complex = y_in

    y_out_complex = np.empty(len(y_in_complex), dtype=complex)

    for i in range(len(x_in)):

        y_out_complex[i] = y_in_complex[i]*np.exp(1j*t*x_in[i])

    y_imag_integrand = np.imag(y_out_complex)

    return y_imag_integrand


def fourier_transform_data_real(T,x_in,y_in):

    val = np.empty(len(T))

    for i in range(len(val)):

        val[i] = fourier_transform_real(T[i],x_test,y_test)

    return val


def fourier_transform_data_imag(T,x_in,y_in):

    val = np.empty(len(T))

    for i in range(len(val)):

        val[i] = fourier_transform_imag(T[i],x_in,y_in)

    return val
