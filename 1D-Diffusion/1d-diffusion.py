from grid import StructuredGrid, UnstructuredGrid
from element import Element

import numpy as np
import matplotlib.pyplot as plt

# Number of elements
nx = 10
# Number of time steps
nt = 1000
# Number of solution points in an element
k = 4

# Interval on x axis
interval = (0.0, 1.0)

# Initial conditions
# ic = lambda x: np.exp(-40 * (x - 0.5)**2)
ic = lambda x: 1.0 - x * x * x

e = Element(4, 1, 1)
e.plotBasisFunctions()
e.plotCorrectionFunctions()
plt.show()

# Solution points, 0 = equidistant, 1 = Gauss, 2 = Lobatto
solutionPoints = 1

# Scheme, 0 = DG, 1 = G2
scheme = 1

# CFL number (0.9 * CFL limit for rk4, complete guess as not included in Huynh 2009)
CFL = 0.4
if k == 3:
    CFL *= 0.145
elif k == 4:
    CFL *= 0.145
elif k == 5:
    CFL *= 0.1
elif k == 6:
    CFL *= 0.05

# Create 1D structured grid and set initial condition
grid = StructuredGrid(interval, nx, k, ic, 1.0, 0.0, solutionPoints, scheme)
dt = (CFL * grid.getdx())

# Create 1D unstructured grid and set initial conditions
# dx = [0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2]
# grid = UnstructuredGrid(0.0, dx, k, ic, 1.0, 0.0, solutionPoints, scheme)
# # dt
# dt = (CFL * min(grid.getdx()))

print("CFL: " + str(CFL))
print("dt: " + str(dt))

plt.figure("Solution Basis")
grid.plotSolution()
grid.plot(dt * nt)
plt.show()

# Advance in time using 4th order Runge-Kutta Method
for i in range(nt):
    grid.rk4Step(dt)

# Plot grid
plt.figure("Solution Basis")
grid.plotSolution()
grid.plot(dt * nt)
plt.show()
