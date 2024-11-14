import numpy as np
import rateMatrix
import DDS
import prettyprint


H_DVR_SINGLE, Q_DVR_SINGLE = DDS.calc_H_DVR_Q_DVR(1, 1.4, 0.05)[0], DDS.calc_H_DVR_Q_DVR(1, 1.4, 0.05)[1]
H_DVR_DOUBLE, Q_DVR_DOUBLE = DDS.calc_H_DVR_Q_DVR(2, 1.4, 0.05)[0], DDS.calc_H_DVR_Q_DVR(2, 1.4, 0.05)[1]
GAMMA_APPROX = rateMatrix.Gamma(1, 1.4, 0.05, 1, 0.097, 0.18, 1)
GAMMA_UNAPPROX = rateMatrix.rate_matrix_unapprox(1, 1.4, 0.05, 1, 0.097, 0.18, 1)
GAMMA_UNAPPROX_DRIVEN = rateMatrix.rate_matrix_unapprox_driven(1, 1.4, 0.05, 1, 0.097, 0.18, 0.8, 0.4, 1)
TEST_RATE_MATRIX = rateMatrix.rate_matrix(1, 1.4, 0.05, 1, 0.097, 0.18, 1)

print('GAMMA_APPROX=')
prettyprint.printA(GAMMA_APPROX)
print('TEST_RATE_MATRIX=')
prettyprint.printA(TEST_RATE_MATRIX)
