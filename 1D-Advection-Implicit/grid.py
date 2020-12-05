from element import Element

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import matplotlib.pyplot as plt

class Grid:

    # Periodic boundary conditions
    def roeFlux(self):
        leftElement = self.leftElement
        for i in range(self.nx):
            rightElement = leftElement.getRightElement()
            ul = leftElement.getRightSolution()
            fl = leftElement.getRightRoeFlux()
            ur = rightElement.getLeftSolution()
            fr = rightElement.getLeftRoeFlux()
            au = self.a
            if ur != ul:
                au = (fr - fl) / (ur - ul)
            # if au >= 0:
            #     fUpwind = fl
            # else:
            #     fUpwind = fr
            fUpwind = 0.5 * (fl + fr) - 0.5 * (abs(au)) * (ur - ul)
            leftElement.setRightUpwindFlux(fUpwind)
            rightElement.setLeftUpwindFlux(fUpwind)
            leftElement = rightElement

    def calculateContinuousFlux(self):
        element = self.leftElement
        for i in range(self.nx):
            element.calculateContinuousFlux()
            element = element.getRightElement()

    def calculateContinuousFluxGradient(self):
        element = self.leftElement
        for i in range(self.nx):
            element.calculateContinuousFluxGradient()
            element = element.getRightElement()

    # March in time using backwards Euler
    def backwardsEuler(self, dt):
        self.dt = dt
        un = self.getGlobalSolution()
        result, errorCode = cg(self.lo, un, tol=1e-12)
        self.setGlobalSolution(result)

    # Calculate (du)/(dx)
    def calcdudx(self):
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()

    # Get (du)/(dx) of all elements
    def getGlobaldudx(self):
        currentElement = self.leftElement
        dudx = []
        for i in range(self.nx):
            dudx.extend(currentElement.getdudx())
            currentElement = currentElement.getRightElement()
        return dudx.copy()

    # v is the latest set of guesses (i.e. a new solution to try)
    def mv(self, u):
        un = self.getGlobalSolution()
        self.setGlobalSolution(u)
        self.calcdudx()
        grad = self.dt * np.array(self.getGlobaldudx())
        self.setGlobalSolution(un)
        return u + grad

    def getdx(self):
        return self.dx

    # Get solution of all elements
    def getGlobalSolution(self):
        currentElement = self.leftElement
        solution = []
        for i in range(self.nx):
            solution.extend(currentElement.getSolution())
            currentElement = currentElement.getRightElement()
        return solution.copy()

    # Set solution of all elements
    def setGlobalSolution(self, u):
        currentElement = self.leftElement
        for i in range(self.nx):
            start = i * self.k
            currentElement.setSolutionPointValues(u[start : start + self.k])
            currentElement = currentElement.getRightElement()

    def plotSolution(self):
        element = self.leftElement
        for i in range(self.nx):
            element.plotLocalSolution()
            element = element.getRightElement()

    def plotDiscontinuousFlux(self):
        element = self.leftElement
        for i in range(self.nx):
            element.plotDiscontinuousFlux()
            element = element.getRightElement()

    def plotContinuousFlux(self):
        element = self.leftElement
        for i in range(self.nx):
            element.plotContinuousFlux()
            element = element.getRightElement()

    def plotLocalContinuousFluxGrad(self):
        element = self.leftElement
        for i in range(self.nx):
            element.plotLocalContinuousFluxGrad()
            element = element.getRightElement()

    def plot(self, t):
        plt.figure("Solution Points")
        currentElement = self.leftElement
        yVal = []
        xVal = []
        # Plot solution
        for i in range(self.nx):
            solution = currentElement.getSolution()
            globalSolutionPoints = currentElement.getGlobalSolutionPoints()
            # Elements share boundaries so don't get right boundary
            for n in range(self.k - 1):
                yVal.append(solution[n])
                xVal.append(globalSolutionPoints[n])
            currentElement = currentElement.getRightElement()
        yValSol = self.ic((np.array(xVal) - self.a * t) % 1.0)
        plt.plot(xVal, yVal)
        plt.plot(xVal, yValSol)

class StructuredGrid(Grid):
    def __init__(self, interval, nx, k, a, flux, ic, solutionPoints, scheme):
        self.nx = nx
        intervalLen = interval[1] - interval[0]
        self.dx = intervalLen / nx
        self.k = k
        self.a = a
        self.fluxFunc = flux
        self.ic = ic
        self.dt = 0.0

        # Generate the required elements
        x = interval[0]
        self.leftElement = Element(self.k, self.dx, x, self.fluxFunc, solutionPoints, scheme)
        self.leftElement.setLeftElement(None)
        # Set initial conditions
        solutionPts = self.leftElement.getGlobalSolutionPoints()
        self.leftElement.setSolutionPointValues(ic(solutionPts))

        prevElement = self.leftElement

        # Construct 1D regular mesh of elements
        for i in range(1, self.nx):
            x = interval[0]+ i * self.dx
            newElement = Element(self.k, self.dx, x, self.fluxFunc, solutionPoints, scheme)
            newElement.setLeftElement(prevElement)
            prevElement.setRightElement(newElement)
            # Set initial conditions
            solutionPts = newElement.getGlobalSolutionPoints()
            newElement.setSolutionPointValues(ic(solutionPts))

            prevElement = newElement

        self.rightElement = prevElement
        # Periodic boundary conditions
        self.leftElement.setLeftElement(self.rightElement)
        self.rightElement.setRightElement(self.leftElement)

        # Construct Linear Operator
        self.lo = LinearOperator((self.nx * self.k, self.nx * self.k), matvec=self.mv)

# class UnstructuredGrid(Grid):
#     def __init__(self, start, dx, k, a, flux, ic, solutionPoints, scheme):
#         self.nx = len(dx)
#         self.dx = dx
#         self.k = k
#         self.a = a
#         self.fluxFunc = flux
#
#         # Generate the required elements
#         x = start
#         self.leftElement = Element(self.k, self.dx[0], x, self.fluxFunc, solutionPoints, scheme)
#         self.leftElement.setLeftElement(None)
#         # Set initial conditions
#         solutionPts = self.leftElement.getGlobalSolutionPoints()
#         self.leftElement.setSolutionPointValues(ic(solutionPts))
#
#         prevElement = self.leftElement
#
#         # Construct 1D regular mesh of elements
#         for i in range(1, self.nx):
#             x += self.dx[i - 1]
#             newElement = Element(self.k, self.dx[i], x, self.fluxFunc, solutionPoints, scheme)
#             newElement.setLeftElement(prevElement)
#             prevElement.setRightElement(newElement)
#             # Set initial conditions
#             solutionPts = newElement.getGlobalSolutionPoints()
#             newElement.setSolutionPointValues(ic(solutionPts))
#
#             prevElement = newElement
#
#         self.rightElement = prevElement
#         # Periodic boundary conditions
#         self.leftElement.setLeftElement(self.rightElement)
#         self.rightElement.setRightElement(self.leftElement)
