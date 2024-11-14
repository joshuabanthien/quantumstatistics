import numpy as np
import scipy as sci
import math


#fits an exponential curve to a set of data 

def EXPONENTIAL_DECAY_FIT(T_DATA, Y_DATA, a0, b0, c0):

    p0 = (a0, b0, c0)

    params, cv = sci.optimize.curve_fit(monoExp, T_DATA, Y_DATA, p0)

    return params
