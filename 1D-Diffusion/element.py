import numpy as np
from numpy.polynomial.polynomial import polyval, polyadd, polyder
import matplotlib.pyplot as plt

from scheme_dg import SchemeDG
from scheme_g2 import SchemeG2

# For now assume 4 points
class Element:
    def __init__(self, k, dx, x, solpoints=2, scheme=0):
        self.left = None
        self.right = None
        self.k = k
        # Centre of element
        self.x = x + dx / 2
        self.dx = dx
        self.solution = np.zeros(k)
        self.setSolutionPoints(solpoints)
        if scheme == 0:
            self.scheme = SchemeDG(self.solutionPts)
        else:
            self.scheme = SchemeG2(self.solutionPts)
        self.k0 = np.zeros(k)
        self.k1 = np.zeros(k)
        self.k2 = np.zeros(k)
        self.k3 = np.zeros(k)
        self.k4 = np.zeros(k)

    def setSolutionPoints(self, solpoints):
        if solpoints == 0:
            # Use equally spaced points
            self.solutionPts = np.linspace(-1.0, 1.0, self.k)
        if solpoints == 1:
            # Use Gauss Points
            if self.k == 1:
                self.solutionPts = np.array([0.0])
            elif self.k == 2:
                self.solutionPts = np.array([-0.57735, 0.57735])
            elif self.k == 3:
                self.solutionPts = np.array([-0.774597, 0.0, 0.774597])
            elif self.k == 4:
                self.solutionPts = np.array([-0.861136, -0.339981, 0.339981, 0.861136])
            elif self.k == 5:
                self.solutionPts = np.array([-0.90618, -0.538469, 0.0, 0.538469, 0.90618])
            else:
                raise Exception("Gauss points above 5th order are not currently implemented")
        if solpoints == 2:
            # Use Lobatto Points
            if self.k == 3:
                self.solutionPts = np.array([-1.0, 0.0, 1.0])
            elif self.k == 4:
                self.solutionPts = np.array([-1.0, -0.447214, 0.447214, 1.0])
            elif self.k == 5:
                self.solutionPts = np.array([-1.0, -0.654654, 0.0, 0.654654, 1.0])
            elif self.k == 6:
                self.solutionPts = np.array([-1.0, -0.765055, -0.285232, 0.285232, 0.765055, 1.0])
            else:
                raise Exception("Lobatto points above 6th order or below 3rd order are not currently implemented")

    def setLeftElement(self, l):
        self.left = l

    def setRightElement(self, r):
        self.right = r

    def getLeftElement(self):
        return self.left

    def getRightElement(self):
        return self.right

    def setSolutionPointValues(self, values):
        for x in range(self.k):
            self.solution[x] = values[x]
        self.solutionPoly = self.scheme.getApproxFunction(self.solution)
        self.solutionGradPoly = polyder(self.solutionPoly)
        self.leftSolution = polyval(-1.0, self.solutionPoly)
        self.rightSolution = polyval(1.0, self.solutionPoly)

    def calculateCommonSolutionPoly(self):
        # correctionFunL = (self.leftCommonSolution - self.leftSolution) * self.scheme.getLeftCorrectionFunction()
        # correctionFunR = (self.rightCommonSolution - self.rightSolution) * self.scheme.getRightCorrectionFunction()
        # self.solutionContinuousPoly = polyadd(self.solutionPoly, correctionFunL)
        # self.solutionContinuousPoly = polyadd(self.solutionContinuousPoly, correctionFunR)
        # self.solutionContinuous = polyval(self.solutionPts, self.solutionContinuousPoly)

        correctionFunGradL = (self.rightCommonSolution - self.leftSolution) * self.scheme.getLeftCorrectionFunctionGrad()
        correctionFunGradR = (self.rightCommonSolution - self.leftSolution) * self.scheme.getRightCorrectionFunctionGrad()
        self.solutionGradContinuousPoly = polyadd(self.solutionGradPoly, correctionFunGradL)
        self.solutionGradContinuousPoly = polyadd(self.solutionGradContinuousPoly, correctionFunGradR)
        self.solutionGradContinuous = (2.0 / self.dx) * polyval(self.solutionPts, self.solutionGradContinuousPoly)
        self.leftGrad = (2.0 / self.dx) * polyval(-1.0, self.solutionGradContinuousPoly)
        self.rightGrad = (2.0 / self.dx) * polyval(1.0, self.solutionGradContinuousPoly)

    def calculateCommonGradPoly(self):
        # correctionFunL = (self.leftCommonGrad - self.leftGrad) * self.scheme.getLeftCorrectionFunction()
        # correctionFunR = (self.rightCommonGrad - self.rightGrad) * self.scheme.getRightCorrectionFunction()
        # self.solutionGradContinuousPoly = polyadd(self.solutionGradContinuousPoly, correctionFunL)
        # self.solutionGradContinuousPoly = polyadd(self.solutionGradContinuousPoly, correctionFunR)
        # self.solutionGradContinuous = polyval(self.solutionPts, self.solutionGradContinuousPoly)

        correctionFunGradL = (self.leftCommonGrad - self.leftGrad) * self.scheme.getLeftCorrectionFunctionGrad()
        correctionFunGradR = (self.rightCommonGrad - self.rightGrad) * self.scheme.getRightCorrectionFunctionGrad()
        self.solutionGrad2Poly = polyadd(polyder(self.solutionGradContinuousPoly), correctionFunGradL)
        self.solutionGrad2Poly = polyadd(self.solutionGrad2Poly, correctionFunGradR)
        self.solutionGrad2 = polyval(self.solutionPts, self.solutionGrad2Poly)

    def getSolution(self):
        return self.solution.copy()

    def getLocalSolutionPoints(self):
        return self.solutionPts.copy()

    def getGlobalSolutionPoints(self):
        return self.x + ((self.solutionPts) * self.dx) / 2

    def getLeftSolution(self):
        return self.leftSolution

    def getRightSolution(self):
        return self.rightSolution

    def getLeftGrad(self):
        return self.leftGrad

    def getRightGrad(self):
        return self.rightGrad

    def setLeftCommonSolution(self, sol):
        self.leftCommonSolution = sol

    def setRightCommonSolution(self, sol):
        self.rightCommonSolution = sol

    def setLeftCommonGrad(self, grad):
        self.leftCommonGrad = grad

    def setRightCommonGrad(self, grad):
        self.rightCommonGrad = grad

    def getdudt(self):
        return self.solutionGrad2.copy()

    # Functions for rk4
    def storeK0(self):
        self.k0 = self.getSolution()

    def storeK1AndUpdate(self, dt):
        self.k1 = self.getdudt()
        self.setSolutionPointValues(self.k0 + (dt / 2.0) * self.k1)

    def storeK2AndUpdate(self, dt):
        self.k2 = self.getdudt()
        self.setSolutionPointValues(self.k0 + dt * (self.k2 / 2))

    def storeK3AndUpdate(self, dt):
        self.k3 = self.getdudt()
        self.setSolutionPointValues(self.k0 + dt * self.k3)

    def storeK4AndUpdate(self, dt):
        self.k4 = self.getdudt()
        vals = self.k0 + (dt / 6.0) * (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)
        self.setSolutionPointValues(vals)

    def plotLocalSolution(self):
        x = np.linspace(-1.0, 1.0, 50)
        y = polyval(x, self.solutionPoly)
        x = np.linspace(self.x - self.dx / 2, self.x + self.dx / 2, 50)
        plt.plot(x, y)

    def plotCorrectionFunctions(self):
        leftCF = self.scheme.getLeftCorrectionFunction()
        rightCF = self.scheme.getRightCorrectionFunction()
        x = np.linspace(-1.0, 1.0, 100)
        ly = polyval(x, leftCF)
        ry = polyval(x, rightCF)
        plt.plot(x, ly)
        plt.plot(x, ry)

    def plotBasisFunctions(self):
        self.scheme.plotBasisFunctions()
