import numpy as np

class Element:
    def __init__(self, k):
        self.left = None
        self.right = None
        self.k = k
        self.solutionPtsVal = np.zeros(k)
        # Use equally spaced points for now
        self.solutionPts = np.linspace(-1.0, 1.0, k)
        # Use Legendre polynomials as basis
        self.basis = np.polynomial.legendre.Legendre.fit(self.solutionPts, self.solutionPtsVal, k)

    def setLeftElement(self, l):
        self.left = l

    def setRightElement(self, r):
        self.right = r

    def setSolutionPointValues(self, values):
        for x in range(self.k):
            self.solutionPtsVal[x] = values[x]
