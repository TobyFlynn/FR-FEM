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
            self.solutionPts = np.array([-0.861136, -0.339981, 0.339981, 0.861136])
        if solpoints == 2:
            # Use Lobatto Points
            self.solutionPts = np.array([-1.0, -0.447214, 0.447214, 1.0])

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

    def getdudt(self):
        return -1.0 * self.getGlobalContinuousFluxGradient()

    # Functions for Roe's Flux
    def getLeftSolution(self):
        return self.leftRoeSolution

    def getRightSolution(self):
        return self.rightRoeSolution

    def getLeftRoeFlux(self):
        # Convert from local to global
        return self.leftRoeFlux

    def getRightRoeFlux(self):
        # Convert from local to global
        return self.rightRoeFlux

    def setLeftUpwindFlux(self, f):
        # Convert from global to local flux
        self.flUpwind = f

    def setRightUpwindFlux(self, f):
        # Convert from global to local flux
        self.frUpwind = f

    # Functions for rk4
    def storeK0(self):
        self.k0 = self.getSolution()

    def storeK1AndUpdate(self, dt):
        self.k1 = self.getdudt().copy()
        self.setSolutionPointValues(self.k0 + (dt / 2.0) * self.k1)

    def storeK2AndUpdate(self, dt):
        self.k2 = self.getdudt().copy()
        self.setSolutionPointValues(self.k0 + dt * (self.k2 / 2))

    def storeK3AndUpdate(self, dt):
        self.k3 = self.getdudt().copy()
        self.setSolutionPointValues(self.k0 + dt * self.k3)

    def storeK4AndUpdate(self, dt):
        self.k4 = self.getdudt().copy()
        vals = self.k0 + (dt / 6.0) * (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)
        self.setSolutionPointValues(vals)

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
