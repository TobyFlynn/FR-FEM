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
        # Periodic boundary conditions
        self.leftElement.setLeftElement(self.rightElement)
        self.rightElement.setRightElement(self.leftElement)

    # Periodic boundary conditions
    def roeFlux(self, a=1):
        leftElement = self.leftElement
        for i in range(self.nx):
            rightElement = leftElement.getRightElement()
            ul = leftElement.getRightSolution()
            fl = leftElement.getRightRoeFlux()
            ur = rightElement.getLeftSolution()
            fr = rightElement.getLeftRoeFlux()
            au = a
            # fUpwind = 0
            if ur != ul:
                au = (fr - fl) / (ur - ul)
            # if au >= 0:
            #     fUpwind = fl
            # else:
            #     fUpwind = fr
            fUpwind = 0.5 * (fl + fr) - 0.5 * (abs(au)) * (ur - ul)
            leftElement.setRightUpwindFlux(fUpwind)
            rightElement.setLeftUpwindFlux(fUpwind)
            # print("fl: " + str(fl) + " fr: " + str(fr) + " fUpwind: " + str(fUpwind) + " fUpwind2: " + str(fUpwind2))
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

    def eulerStep(self, dt):
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()
        plt.figure("Euler")
        self.plotLocalDiscontinuousFlux()
        self.plotLocalContinuousFlux()
        # Update solution using Euler time marching
        element = self.leftElement
        for i in range(self.nx):
            element.setSolutionPointValues(element.getSolution() + dt * element.getdudt())
            element = element.getRightElement()

    def rk4Step(self, dt):
        # Store initial solution in k0
        self.storeK0()
        plt.figure("ICs")
        self.plotSolution()

        # Calculate k1, store k1 and update solution
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()
        plt.figure("K1")
        self.plotLocalDiscontinuousFlux()
        self.plotLocalContinuousFlux()
        self.plotSolution()
        self.plotLocalContinuousFluxGrad()
        self.storeK1AndUpdate(dt)

        # Calculate k2, store k2 and update solution
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()
        plt.figure("K2")
        self.plotLocalDiscontinuousFlux()
        self.plotLocalContinuousFlux()
        self.plotSolution()
        self.plotLocalContinuousFluxGrad()
        self.storeK2AndUpdate(dt)

        # Calculate k3, store k3 and update solution
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()
        plt.figure("K3")
        self.plotLocalDiscontinuousFlux()
        self.plotLocalContinuousFlux()
        self.plotSolution()
        self.plotLocalContinuousFluxGrad()
        self.storeK3AndUpdate(dt)

        # Calculate k4, store k4 and update solution
        self.roeFlux()
        self.calculateContinuousFlux()
        self.calculateContinuousFluxGradient()
        plt.figure("K4")
        self.plotLocalDiscontinuousFlux()
        self.plotLocalContinuousFlux()
        self.plotSolution()
        self.plotLocalContinuousFluxGrad()
        self.storeK4AndUpdate(dt)

    def getdx(self):
        return self.dx

    def plotSolution(self):
        element = self.leftElement
        for i in range(self.nx):
            element.plotLocalSolution()
            element = element.getRightElement()

    def plotLocalDiscontinuousFlux(self):
        element = self.leftElement
        for i in range(self.nx):
            element.plotLocalDiscontinuousFlux()
            element = element.getRightElement()

    def plotLocalContinuousFlux(self):
        element = self.leftElement
        for i in range(self.nx):
            element.plotLocalContinuousFlux()
            element = element.getRightElement()

    def plotLocalContinuousFluxGrad(self):
        element = self.leftElement
        for i in range(self.nx):
            element.plotLocalContinuousFluxGrad()
            element = element.getRightElement()

    def plot(self):
        plt.figure("Solution")
        currentElement = self.leftElement
        yVal = []
        xVal = []
        # fluxVal = []
        # fluxGrad = []
        # Plot solution
        for i in range(self.nx):
            solution = currentElement.getSolution()
            # flux = currentElement.getGlobalFlux()
            # fluxG = currentElement.getGlobalFluxGradient()
            x = i * self.dx
            # Elements share boundaries so don't get right boundary
            for n in range(self.k - 1):
                yVal.append(solution[n])
                xVal.append(x + n * (self.dx / (self.k - 1)))
                # fluxVal.append(flux[n])
                # fluxGrad.append(fluxG[n])

            currentElement = currentElement.getRightElement()

        plt.plot(xVal, yVal)
        # plt.plot(xVal, fluxVal)
        # plt.plot(xVal, fluxGrad)
        plt.show()
