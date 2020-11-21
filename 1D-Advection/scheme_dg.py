import numpy as np
from numpy.polynomial.polynomial import polyval, polyadd, polyder
import matplotlib.pyplot as plt

from scheme import Scheme

# For now assume 4 points
class SchemeDG(Scheme):

    def __init__(self, points):
        super().__init__(points)
        coeff = [0] * (self.k - 1)
        coeff.append(-1)
        coeff.append(1)
        self.leftCorrectionFunction = np.polynomial.legendre.leg2poly(((-1) ** self.k) * 0.5 * np.array(coeff))
        self.leftCorrectionFunctionGrad = polyder(self.leftCorrectionFunction)
        coeff = [0] * (self.k - 1)
        coeff.append(1)
        coeff.append(1)
        self.rightCorrectionFunction = np.polynomial.legendre.leg2poly((0.5 * np.array(coeff)))
        self.rightCorrectionFunctionGrad = polyder(self.rightCorrectionFunction)
