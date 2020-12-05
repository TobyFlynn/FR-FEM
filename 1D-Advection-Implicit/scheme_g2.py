import numpy as np
from numpy.polynomial.polynomial import polyval, polyadd, polyder

from scheme import Scheme

class SchemeG2(Scheme):

    def __init__(self, points):
        super().__init__(points)
        coeff = [0] * (self.k - 1)
        coeff.append(-1)
        coeff.append(1)
        self.leftCorrectionFunction = ((self.k - 1) / (2 * self.k - 1)) * np.polynomial.legendre.leg2poly((((-1) ** self.k) * 0.5 * np.array(coeff)))
        coeff = [0] * (self.k - 2)
        coeff.append(-1)
        coeff.append(1)
        self.leftCorrectionFunction = polyadd(self.leftCorrectionFunction, (self.k / (2 * self.k - 1)) * np.polynomial.legendre.leg2poly((((-1) ** (self.k - 1)) * 0.5 * np.array(coeff))))
        self.leftCorrectionFunctionGrad = polyder(self.leftCorrectionFunction)

        coeff = [0] * (self.k - 1)
        coeff.append(1)
        coeff.append(1)
        self.rightCorrectionFunction = ((self.k - 1) / (2 * self.k - 1)) * np.polynomial.legendre.leg2poly((0.5 * np.array(coeff)))
        coeff = [0] * (self.k - 2)
        coeff.append(1)
        coeff.append(1)
        self.rightCorrectionFunction = polyadd(self.rightCorrectionFunction, (self.k / (2 * self.k - 1)) * np.polynomial.legendre.leg2poly((0.5 * np.array(coeff))))
        self.rightCorrectionFunctionGrad = polyder(self.rightCorrectionFunction)
