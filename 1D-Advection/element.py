import numpy as np

class Element:
    def __init__(self, k, dx, x):
        self.left = None
        self.right = None
        self.k = k
        self.x = x
        self.dx = dx
        self.solutionPtsVal = np.zeros(k)
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
            self.solutionPtsVal[x] = values[x]
        #self.updateBasis()

    def updateBasis(self):
        self.basis = np.polynomial.legendre.Legendre.fit(self.solutionPts, self.solutionPtsVal, self.k).convert().coef

    def getSolution(self):
        #return np.polynomial.legendre.legval(self.solutionPts, self.basis)
        return self.solutionPtsVal

    def getLocalSolutionPoints(self):
        return self.solutionPts

    def getGlobalSolutionPoints(self):
        # Shift solution points from [-1,1] to [0,2]
        # Then scale to [0,1] and multiply by dx to scale to [0,dx]
        # Then add x to get global coordinates
        return self.x + ((1.0 + self.solutionPts) / 2) * self.dx
