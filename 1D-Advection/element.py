import numpy as np

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
        # Use equally spaced points for now
        self.solutionPts = np.linspace(-1.0, 1.0, k)

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

    def updateBasis(self):
        self.solutionBasis = np.polynomial.legendre.legfit(self.solutionPts, self.solution, self.k - 1)
        self.solutionGrad = np.polynomial.legendre.legder(self.solutionBasis)

    def updateFlux(self, a=1):
        self.flux = np.polynomial.legendre.legval(self.solutionPts, a * self.solutionGrad)
        self.updateFluxBasis()

    def updateFluxBasis(self):
        self.fluxBasis = np.polynomial.legendre.legfit(self.solutionPts, self.flux, self.k - 1)
        self.fluxGrad = np.polynomial.legendre.legder(self.fluxBasis)

    def getSolution(self):
        return np.polynomial.legendre.legval(self.solutionPts, self.solutionBasis)
        #return self.solution

    def getLocalSolutionPoints(self):
        return self.solutionPts

    def getGlobalSolutionPoints(self):
        return self.x + ((self.solutionPts) * self.dx) / 2

    def getLocalSolutionGradient(self):
        return np.polynomial.legendre.legval(self.solutionPts, self.solutionGrad)

    def getGlobalSolutionGradient(self):
        return (2.0 / self.dx) * self.getLocalSolutionGradient()

    # For now assume a = 1
    def getLocalFlux(self):
        return self.flux

    def getGlobalFlux(self):
        return (2.0 / self.dx) * self.flux

    def getLocalFluxGradient(self):
        return np.polynomial.legendre.legval(self.solutionPts, self.fluxGrad)

    def getGlobalFluxGradient(self):
        return (2.0 / self.dx) * self.getLocalFluxGradient()

    # Functions for Roe's Flux
    def getLeftSolution(self):
        return self.getSolution()[0]

    def getRightSolution(self):
        return self.getSolution()[self.k - 1]

    def getLeftFlux(self):
        return getGlobalFlux()[0]

    def getRightFlux(self):
        return getGlobalFlux()[self.k - 1]

    def setLeftUpwindFlux(self, f):
        # Convert from global to local flux
        self.flUpwind = (self.dx / 2.0) * f

    def setRightUpwindFlux(self, f):
        # Convert from global to local flux
        self.frUpwind = (self.dx / 2.0) * f
