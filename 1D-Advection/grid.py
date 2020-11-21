from element import Element

import numpy as np
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

    def storeK0(self):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK0()
            element = element.getRightElement()

    def storeK1AndUpdate(self, dt):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK1AndUpdate(dt)
            element = element.getRightElement()

    def storeK2AndUpdate(self, dt):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK2AndUpdate(dt)
            element = element.getRightElement()

    def storeK3AndUpdate(self, dt):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK3AndUpdate(dt)
            element = element.getRightElement()

    def storeK4AndUpdate(self, dt):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK4AndUpdate(dt)
            element = element.getRightElement()

    # Check whether only need to consdier tn + h/2 for boundary conditions
    def rk4Step(self, dt):
        # Store initial solution in k0
        self.storeK0()

        # Calculate k1, store k1 and update solution
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()
        self.storeK1AndUpdate(dt)

        # Calculate k2, store k2 and update solution
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()
        self.storeK2AndUpdate(dt)

        # Calculate k3, store k3 and update solution
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()
        self.storeK3AndUpdate(dt)

        # Calculate k4, store k4 and update solution
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()
        self.storeK4AndUpdate(dt)
        # plt.figure("RK4")
        # self.plotContinuousFlux()
        # self.plotLocalContinuousFluxGrad()

    def getdx(self):
        return self.dx

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

    def plot(self):
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
        plt.plot(xVal, yVal)

class StructuredGrid(Grid):
    def __init__(self, interval, nx, k, a, flux, ic, solutionPoints, scheme):
        self.nx = nx
        intervalLen = interval[1] - interval[0]
        self.dx = intervalLen / nx
        self.k = k
        self.a = a
        self.fluxFunc = flux

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

class UnstructuredGrid(Grid):
    def __init__(self, start, dx, k, a, flux, ic, solutionPoints, scheme):
        self.nx = len(dx)
        self.dx = dx
        self.k = k
        self.a = a
        self.fluxFunc = flux

        # Generate the required elements
        x = start
        self.leftElement = Element(self.k, self.dx[0], x, self.fluxFunc, solutionPoints, scheme)
        self.leftElement.setLeftElement(None)
        # Set initial conditions
        solutionPts = self.leftElement.getGlobalSolutionPoints()
        self.leftElement.setSolutionPointValues(ic(solutionPts))

        prevElement = self.leftElement

        # Construct 1D regular mesh of elements
        for i in range(1, self.nx):
            x += self.dx[i - 1]
            newElement = Element(self.k, self.dx[i], x, self.fluxFunc, solutionPoints, scheme)
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
