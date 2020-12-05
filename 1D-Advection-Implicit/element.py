import numpy as np
from numpy.polynomial.polynomial import polyval, polyadd, polyder
import matplotlib.pyplot as plt

from scheme_dg import SchemeDG
from scheme_g2 import SchemeG2

# For now assume 4 points
class Element:
    def __init__(self, k, dx, x, fluxFunc, solpoints=2, scheme=0):
        self.left = None
        self.right = None
        self.k = k
        # Centre of element
        self.x = x + dx / 2
        self.dx = dx
        self.fluxFunc = fluxFunc
        self.solution = np.zeros(k)
        self.flux = np.zeros(k)
        self.fluxGrad = np.zeros(k)
        self.fluxContinuous = np.zeros(k)
        self.fluxContinuousGrad = np.zeros(k)
        self.setSolutionPoints(solpoints)
        if scheme == 0:
            self.scheme = SchemeDG(self.solutionPts)
        else:
            self.scheme = SchemeG2(self.solutionPts)

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
        self.updateSolutionPoly()
        self.updateFlux()

    def updateSolutionPoly(self):
        self.solutionPoly = self.scheme.getApproxFunction(self.solution)
        self.leftRoeSolution = polyval(-1.0, self.solutionPoly)
        self.rightRoeSolution = polyval(1.0, self.solutionPoly)
        self.leftRoeFlux = self.fluxFunc(polyval(-1.0, self.solutionPoly))
        self.rightRoeFlux = self.fluxFunc(polyval(1.0, self.solutionPoly))

    def updateFlux(self):
        self.flux = self.fluxFunc(self.solution)
        self.updateFluxPoly()
        self.fluxGrad = polyval(self.solutionPts, self.fluxGradPoly)

    def updateFluxPoly(self):
        self.fluxPoly = self.scheme.getApproxFunction(self.flux)
        self.fluxGradPoly = polyder(self.fluxPoly)

    # Not a necessary calculation but included just for completeness
    def calculateContinuousFlux(self):
        correctionFunL = (self.flUpwind - self.leftRoeFlux) * self.scheme.getLeftCorrectionFunction()
        correctionFunR = (self.frUpwind - self.rightRoeFlux) * self.scheme.getRightCorrectionFunction()
        self.fluxContinuousPoly = polyadd(self.fluxPoly, correctionFunL)
        self.fluxContinuousPoly = polyadd(self.fluxContinuousPoly, correctionFunR)
        self.fluxContinuous = polyval(self.solutionPts, self.fluxContinuousPoly)

    def calculateContinuousFluxGradient(self):
        correctionFunGradL = (self.flUpwind - self.leftRoeFlux) * self.scheme.getLeftCorrectionFunctionGrad()
        correctionFunGradR = (self.frUpwind - self.rightRoeFlux) * self.scheme.getRightCorrectionFunctionGrad()
        self.fluxContinuousGradPoly = polyadd(self.fluxGradPoly, correctionFunGradL)
        self.fluxContinuousGradPoly = polyadd(self.fluxContinuousGradPoly, correctionFunGradR)
        self.fluxContinuousGrad = polyval(self.solutionPts, self.fluxContinuousGradPoly)

    def getSolution(self):
        return self.solution.copy()

    def getLocalSolutionPoints(self):
        return self.solutionPts.copy()

    def getGlobalSolutionPoints(self):
        return self.x + ((self.solutionPts) * self.dx) / 2

    def getFlux(self):
        return self.flux.copy()

    def getLocalFluxGradient(self):
        return self.fluxGrad.copy()

    def getGlobalFluxGradient(self):
        return (2.0 / self.dx) * self.getLocalFluxGradient()

    def getContinuousFlux(self):
        return self.fluxContinuous.copy()

    def getLocalContinuousFluxGradient(self):
        return self.fluxContinuousGrad.copy()

    def getGlobalContinuousFluxGradient(self):
        return (2.0 / self.dx) * self.getLocalContinuousFluxGradient()

    def getdudx(self):
        return self.getGlobalContinuousFluxGradient()

    def getdudt(self):
        return -1.0 * self.getGlobalContinuousFluxGradient()

    # Functions for Roe's Flux
    def getLeftSolution(self):
        return self.leftRoeSolution

    def getRightSolution(self):
        return self.rightRoeSolution

    def getLeftRoeFlux(self):
        return self.leftRoeFlux

    def getRightRoeFlux(self):
        return self.rightRoeFlux

    def setLeftUpwindFlux(self, f):
        self.flUpwind = f

    def setRightUpwindFlux(self, f):
        self.frUpwind = f

    def plotLocalSolution(self):
        x = np.linspace(-1.0, 1.0, 50)
        y = polyval(x, self.solutionPoly)
        x = np.linspace(self.x - self.dx / 2, self.x + self.dx / 2, 50)
        plt.plot(x, y)

    def plotDiscontinuousFlux(self):
        x = np.linspace(-1.0, 1.0, 50)
        y = polyval(x, self.fluxPoly)
        x = np.linspace(self.x - self.dx / 2, self.x + self.dx / 2, 50)
        plt.plot(x, y)

    def plotContinuousFlux(self):
        x = np.linspace(-1.0, 1.0, 50)
        y = polyval(x, self.fluxContinuousPoly)
        x = np.linspace(self.x - self.dx / 2, self.x + self.dx / 2, 50)
        plt.plot(x, y)

    def plotLocalContinuousFluxGrad(self):
        x = np.linspace(-1.0, 1.0, 50)
        y = polyval(x, self.fluxContinuousGradPoly)
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
