import numpy as np
from scipy.interpolate import lagrange

# For now assume 4 points
class Basis:

    def __init__(self, ptns):
        self.ptns = ptns
        self.k = self.ptns.shape[0]
        coeff = [0] * (self.k - 1)
        coeff.append(-1)
        coeff.append(1)
        self.leftCorrectionFunction = np.polynomial.legendre.leg2poly(((-1) ** self.k) * 0.5 * np.array(coeff))
        self.leftCorrectionFunctionGrad = np.polynomial.polynomial.polyder(self.leftCorrectionFunction)
        coeff = [0] * (self.k - 1)
        coeff.append(1)
        coeff.append(1)
        self.rightCorrectionFunction = np.polynomial.legendre.leg2poly((0.5 * np.array(coeff)))
        self.rightCorrectionFunctionGrad = np.polynomial.polynomial.polyder(self.rightCorrectionFunction)


    def getBasis(self, values):
        # Convert poly1d that lagrange returns into numpy polynomial
        # TODO make more efficient, don't need to interpolate each time
        # can just calculate basis functions at start then multiply these by
        # the values
        return np.flipud(lagrange(self.ptns, values))

    def getLeftCorrectionFunction(self):
        return self.leftCorrectionFunction.copy()

    def getRightCorrectionFunction(self):
        return self.rightCorrectionFunction.copy()

    def getLeftCorrectionFunctionGrad(self):
        return self.leftCorrectionFunctionGrad.copy()

    def getRightCorrectionFunctionGrad(self):
        return self.rightCorrectionFunctionGrad.copy()
