import numpy as np
from numpy.polynomial.polynomial import polyval, polyadd
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

class Scheme:
    def __init__(self, points):
        self.points = points
        self.k = self.points.shape[0]
        # Construct basis functions
        values = np.zeros(self.k)
        self.basis = []
        for i in range(self.k):
            values[i] = 1.0
            # Convert poly1d that lagrange returns into numpy polynomial
            self.basis.append(np.flipud(lagrange(self.points, values)))
            values[i] = 0.0

    def getApproxFunction(self, values):
        approx = values[0] * self.basis[0]
        for i in range(1, self.k):
            approx = polyadd(approx, values[i] * self.basis[i])
        return approx

    def getLeftCorrectionFunction(self):
        return self.leftCorrectionFunction.copy()

    def getRightCorrectionFunction(self):
        return self.rightCorrectionFunction.copy()

    def getLeftCorrectionFunctionGrad(self):
        return self.leftCorrectionFunctionGrad.copy()

    def getRightCorrectionFunctionGrad(self):
        return self.rightCorrectionFunctionGrad.copy()

    def plotBasisFunctions(self):
        x = np.linspace(-1.0, 1.0, 100)
        for i in range(self.k):
            y = polyval(x, self.basis[i])
            plt.plot(x, y)
