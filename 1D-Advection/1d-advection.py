from grid import StructuredGrid, UnstructuredGrid
from element import Element

import numpy as np
import matplotlib.pyplot as plt

# Number of elements
nx = 10
# Number of time steps
nt = 500
# Number of solution points in an element
k = 4
# Wave speed
a = 1.0

# Interval on x axis
interval = (0.0, 1.0)

# Initial conditions
ic = lambda x: np.exp(-40 * (x - 0.5)**2)

# Flux function for linear advection
fluxFunc = lambda x, a=a: a * x

# Flux function for the inviscid Burgers' Equation
# fluxFunc = lambda x, a=a: a * (x * x) * 0.5

e = Element(4, 1, 1, fluxFunc)
e.plotBasisFunctions()
e.plotCorrectionFunctions()
plt.show()

# Solution points, 0 = equidistant, 1 = Gauss, 2 = Lobatto
solutionPoints = 2

# Scheme, 0 = DG, 1 = G2
scheme = 0

# CFL number (0.9 * CFL limit for rk4, only exact for k = 4, others are a guess)
CFL = 0.9
if k == 3:
    CFL *= 0.145
elif k == 4:
    CFL *= 0.145
elif k == 5:
    CFL *= 0.1
elif k == 6:
    CFL *= 0.05

# Create 1D structured grid and set initial condition
grid = StructuredGrid(interval, nx, k, a, fluxFunc, ic, solutionPoints, scheme)
dt = (CFL * grid.getdx()) / abs(a)

# Create 1D unstructured grid and set initial conditions
# dx = [0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2]
# grid = UnstructuredGrid(0.0, dx, k, a, fluxFunc, ic, solutionPoints, scheme)
# dt
# dt = (CFL * min(grid.getdx())) / abs(a)

print("CFL: " + str(CFL))
print("dt: " + str(dt))

# Advance in time using 4th order Runge-Kutta Method
for i in range(nt):
    grid.rk4Step(dt)

# Plot grid
plt.figure("Solution Basis")
grid.plotSolution()
grid.plot(dt * nt)
plt.show()
