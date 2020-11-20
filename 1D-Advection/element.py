import numpy as np
import matplotlib.pyplot as plt

class Element:
    def __init__(self, k, dx, x):
        self.left = None
        self.right = None
        self.k = k
        # Centre of element
        self.x = x + dx / 2
        self.dx = dx
        self.solution = np.zeros(k)
        self.flux = np.zeros(k)
        self.fluxGrad = np.zeros(k)
        self.fluxContinuous = np.zeros(k)
        self.fluxContinuousGrad = np.zeros(k)
        # Use equally spaced points for now
        self.solutionPts = np.linspace(-1.0, 1.0, k)
        # rk4 data
        self.k0 = np.zeros(k)
        self.k1 = np.zeros(k)
        self.k2 = np.zeros(k)
        self.k3 = np.zeros(k)
        self.k4 = np.zeros(k)

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
        self.updateBasis()
        self.updateFlux()

    def updateBasis(self, a=1):
        self.solutionBasis = ((self.solution[0] * self.getBasis0()) + (self.solution[1] * self.getBasis1()) +
                              (self.solution[2] * self.getBasis2()) + (self.solution[3] * self.getBasis3()))
        self.solutionGradBasis = np.polynomial.legendre.legder(self.solutionBasis)
        self.leftRoeFlux = a * np.polynomial.legendre.legval(-1, self.solutionGradBasis)
        self.rightRoeFlux = a * np.polynomial.legendre.legval(1, self.solutionGradBasis)
        self.leftRoeSolution = np.polynomial.legendre.legval(-1, self.solutionBasis)
        self.rightRoeSolution = np.polynomial.legendre.legval(1, self.solutionBasis)

    def updateFlux(self, a=1):
        self.flux = np.polynomial.legendre.legval(self.solutionPts, a * self.solutionGradBasis)
        self.updateFluxBasis()
        self.fluxGrad = np.polynomial.legendre.legval(self.solutionPts, a * self.fluxGradBasis)

    def updateFluxBasis(self):
        # self.fluxBasis = np.polynomial.legendre.legfit(self.solutionPts, self.flux, self.k - 1)
        self.fluxBasis = ((self.flux[0] * self.getBasis0()) + (self.flux[1] * self.getBasis1()) +
                          (self.flux[2] * self.getBasis2()) + (self.flux[3] * self.getBasis3()))
        self.fluxGradBasis = np.polynomial.legendre.legder(self.fluxBasis)

    def calculateContinuousFlux(self):
        correctionFunL = (self.flUpwind - self.flux[0]) * self.getLeftCorrectionFunction()
        correctionFunR = (self.frUpwind - self.flux[self.k - 1]) * self.getRightCorrectionFunction()
        self.fluxContinuousBasis = np.polynomial.legendre.legadd(self.fluxBasis, correctionFunL)
        self.fluxContinuousBasis = np.polynomial.legendre.legadd(self.fluxContinuousBasis, correctionFunR)
        self.fluxContinuous = np.polynomial.legendre.legval(self.solutionPts, self.fluxContinuousBasis)

    def calculateContinuousFluxGradient(self):
        correctionFunGradL = (self.flUpwind - self.flux[0]) * self.getLeftCorrectionFunctionGrad()
        correctionFunGradR = (self.frUpwind - self.flux[self.k - 1]) * self.getRightCorrectionFunctionGrad()
        self.fluxContinuousGradBasis = np.polynomial.legendre.legadd(self.fluxGradBasis, correctionFunGradL)
        self.fluxContinuousGradBasis = np.polynomial.legendre.legadd(self.fluxContinuousGradBasis, correctionFunGradR)
        self.fluxContinuousGrad = np.polynomial.legendre.legval(self.solutionPts, self.fluxContinuousGradBasis)

    def getSolution(self):
        #return np.polynomial.legendre.legval(self.solutionPts, self.solutionBasis)
        return self.solution

    def getLocalSolutionPoints(self):
        return self.solutionPts

    def getGlobalSolutionPoints(self):
        return self.x + ((self.solutionPts) * self.dx) / 2

    def getLocalSolutionGradient(self):
        return np.polynomial.legendre.legval(self.solutionPts, self.solutionGradBasis)

    def getGlobalSolutionGradient(self):
        return (2.0 / self.dx) * self.getLocalSolutionGradient()

    def getLocalFlux(self):
        return self.flux

    def getGlobalFlux(self):
        return (2.0 / self.dx) * self.flux

    def getLocalFluxGradient(self):
        return np.polynomial.legendre.legval(self.solutionPts, self.fluxGradBasis)

    def getGlobalFluxGradient(self):
        return (2.0 / self.dx) * self.getLocalFluxGradient()

    def getLocalContinuousFlux(self):
        return self.fluxContinuous

    def getGlobalContinuousFlux(self):
        return (2.0 / self.dx) * self.getLocalContinuousFlux()

    def getLocalContinuousFluxGradient(self):
        return self.fluxContinuousGrad

    def getGlobalContinuousFluxGradient(self):
        return (2.0 / self.dx) * self.getLocalContinuousFluxGradient()

    def getdudt(self):
        return -1.0 * self.getGlobalContinuousFluxGradient()

    # Basis functions (Legendre Polynomials)
    def getBasis0(self):
        yVals = np.array([1.0, 0.0, 0.0, 0.0])
        return np.polynomial.legendre.legfit(self.solutionPts, yVals, self.k - 1)

    def getBasis1(self):
        yVals = np.array([0.0, 1.0, 0.0, 0.0])
        return np.polynomial.legendre.legfit(self.solutionPts, yVals, self.k - 1)

    def getBasis2(self):
        yVals = np.array([0.0, 0.0, 1.0, 0.0])
        return np.polynomial.legendre.legfit(self.solutionPts, yVals, self.k - 1)

    def getBasis3(self):
        yVals = np.array([0.0, 0.0, 0.0, 1.0])
        return np.polynomial.legendre.legfit(self.solutionPts, yVals, self.k - 1)

    # Functions for Roe's Flux
    def getLeftSolution(self):
        return self.leftRoeSolution

    def getRightSolution(self):
        return self.rightRoeSolution

    def getLeftRoeFlux(self, a=1):
        return a * (2.0 / self.dx) * self.leftRoeFlux

    def getRightRoeFlux(self, a=1):
        return a * (2.0 / self.dx) * self.rightRoeFlux

    def setLeftUpwindFlux(self, f):
        # Convert from global to local flux
        self.flUpwind = (self.dx / 2.0) * f

    def setRightUpwindFlux(self, f):
        # Convert from global to local flux
        self.frUpwind = (self.dx / 2.0) * f

    # Functions for rk4
    def storeK0(self):
        self.k0 = self.solution

    def storeK1AndUpdate(self, dt):
        self.k1 = self.getdudt()
        # self.setSolutionPointValues(self.k0 + dt * (self.k1 / 2))
        self.setSolutionPointValues(self.solution + dt * (self.k1 / 2))

    def storeK2AndUpdate(self, dt):
        self.k2 = self.getdudt()
        # self.setSolutionPointValues(self.k0 + dt * (self.k2 / 2))
        self.setSolutionPointValues(self.solution + dt * (self.k2 / 2))

    def storeK3AndUpdate(self, dt):
        self.k3 = self.getdudt()
        # self.setSolutionPointValues(self.k0 + dt * self.k3)
        self.setSolutionPointValues(self.solution + dt * self.k3)

    def storeK4AndUpdate(self, dt):
        self.k4 = self.getdudt()
        vals = self.k0 + (1.0 / 6.0) * dt * (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)
        self.setSolutionPointValues(vals)

    # Correction Functions (Radau Polynomials)
    # Return coefficients for Legendre series
    def getLeftCorrectionFunction(self):
        coeff = [0] * self.k
        coeff.append(1)
        coeff.append(-1)
        return ((-1) ** self.k) * 0.5 * np.array(coeff)

    def getRightCorrectionFunction(self):
        coeff = [0] * self.k
        coeff.append(1)
        coeff.append(1)
        return 0.5 * np.array(coeff)

    def getLeftCorrectionFunctionGrad(self):
        coeff = [0] * self.k
        coeff.append(1)
        coeff.append(-1)
        return np.polynomial.legendre.legder(((-1) ** self.k) * 0.5 * np.array(coeff))

    def getRightCorrectionFunctionGrad(self):
        coeff = [0] * self.k
        coeff.append(1)
        coeff.append(1)
        return np.polynomial.legendre.legder(0.5 * np.array(coeff))

    # Helper functions to plot the various functions
    def plotBasisFunctions(self):
        x = np.linspace(-1.0, 1.0, 50)
        y0 = np.polynomial.legendre.legval(x, self.getBasis0())
        y1 = np.polynomial.legendre.legval(x, self.getBasis1())
        y2 = np.polynomial.legendre.legval(x, self.getBasis2())
        y3 = np.polynomial.legendre.legval(x, self.getBasis3())
        plt.plot(x, y0)
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.plot(x, y3)

    def plotLocalSolution(self):
        x = np.linspace(-1.0, 1.0, 50)
        y = np.polynomial.legendre.legval(x, self.solutionBasis)
        x = np.linspace(self.x - self.dx / 2, self.x + self.dx / 2, 50)
        plt.plot(x, y)

    def plotLocalDiscontinuousFlux(self):
        x = np.linspace(-1.0, 1.0, 50)
        y = np.polynomial.legendre.legval(x, self.fluxBasis)
        x = np.linspace(self.x - self.dx / 2, self.x + self.dx / 2, 50)
        plt.plot(x, y)

    def plotLocalContinuousFlux(self):
        x = np.linspace(-1.0, 1.0, 50)
        y = np.polynomial.legendre.legval(x, self.fluxContinuousBasis)
        x = np.linspace(self.x - self.dx / 2, self.x + self.dx / 2, 50)
        plt.plot(x, y)

    def plotLocalContinuousFluxGrad(self):
        x = np.linspace(-1.0, 1.0, 50)
        y1 = np.polynomial.legendre.legval(x, np.polynomial.legendre.legder(self.fluxContinuousBasis))
        y2 = np.polynomial.legendre.legval(x, self.fluxContinuousGradBasis)
        x = np.linspace(self.x - self.dx / 2, self.x + self.dx / 2, 50)
        plt.plot(x, y1)
        plt.plot(x, y2)

    def plotCorrectionFunctions(self):
        leftCF = self.getLeftCorrectionFunction()
        rightCF = self.getRightCorrectionFunction()
        x = np.linspace(-1.0, 1.0, 100)
        ly = np.polynomial.legendre.legval(x, leftCF)
        ry = np.polynomial.legendre.legval(x, rightCF)
        plt.plot(x, ly)
        plt.plot(x, ry)
