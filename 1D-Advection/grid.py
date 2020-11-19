from element import Element

import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, interval, nx, k, ic):
        self.leftBoundary = interval[0]
        self.rightBoundary = interval[1]
        self.nx = nx
        intervalLen = interval[1] - interval[0]
        self.dx = intervalLen / nx
        self.k = k

        # Generate the required elements
        x = self.leftBoundary
        self.leftElement = Element(self.k, self.dx, x)
        self.leftElement.setLeftElement(None)
        # Set initial conditions
        solutionPts = self.leftElement.getGlobalSolutionPoints()
        self.leftElement.setSolutionPointValues(ic(solutionPts))

        prevElement = self.leftElement

        # Construct 1D regular mesh of elements
        for i in range(1, self.nx):
            x = self.leftBoundary + i * self.dx
            newElement = Element(self.k, self.dx, x)
            newElement.setLeftElement(prevElement)
            prevElement.setRightElement(newElement)
            # Set initial conditions
            solutionPts = newElement.getGlobalSolutionPoints()
            newElement.setSolutionPointValues(ic(solutionPts))

            prevElement = newElement

        self.rightElement = prevElement

    # Check whether using correct flux here
    def roeFlux(self, a=1):
        leftElement = self.leftElement
        for i in range(nx - 1):
            rightElement = leftElement.getRightElement()
            ul = leftElement.getRightSolution()
            fl = leftElement.getRightFlux()
            ur = rightElement.getLeftSolution()
            fr = rightElement.getLeftFlux()
            au = a
            if ur != ul:
                au = (fr - fl) / (ur - ul)
            fUpwind = 0.5 * (fl + fr) - 0.5 * (abs(au)) * (ur - ul)

    def plot(self):
        currentElement = self.leftElement
        yVal = []
        xVal = []
        fluxVal = []
        fluxGrad = []
        # Plot solution
        for i in range(self.nx):
            solution = currentElement.getSolution()
            flux = currentElement.getGlobalFlux()
            fluxG = currentElement.getGlobalFluxGradient()
            x = i * self.dx
            # Elements share boundaries so don't get right boundary
            for n in range(self.k - 1):
                yVal.append(solution[n])
                xVal.append(x + n * (self.dx / (self.k - 1)))
                fluxVal.append(flux[n])
                fluxGrad.append(fluxG[n])

            currentElement = currentElement.getRightElement()

        plt.plot(xVal, yVal)
        plt.plot(xVal, fluxVal)
        plt.plot(xVal, fluxGrad)
        plt.show()
