import DDS
import helpful_functions as hf
import numpy as np
import rateMatrix

print(rateMatrix.Gamma(1, 1.4, 0.05, 1, 0.097, 0.18, 0.5))
print(rateMatrix.Gamma(2, 1.4, 0.05, 1, 0.097, 0.18, 0.5))
print(DDS.calc_H_DVR_Q_DVR(1, 1.4, 0.05)[0])
print(DDS.calc_H_DVR_Q_DVR(2, 1.4, 0.05)[0])
print(DDS.calc_H_DVR_Q_DVR(1, 1.4, 0.05)[1])
print(DDS.calc_H_DVR_Q_DVR(2, 1.4, 0.05)[1])
