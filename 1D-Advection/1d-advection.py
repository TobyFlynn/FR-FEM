from grid import Grid
from element import Element

import numpy as np
import matplotlib.pyplot as plt

# Number of elements
nx = 10
# Number of time steps
nt = 15
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
e.basis.plotBasisFunctions()
e.plotCorrectionFunctions()
plt.show()

# Create 1D regular grid and set initial condition
grid = Grid(interval, nx, k, a, fluxFunc, ic)
# CFL number (0.9 * CFL limit for rk4)
CFL = 0.9 * 0.145
# dt
dt = (CFL * grid.getdx()) / abs(a)

print("CFL: " + str(CFL))
print("dt: " + str(dt))

# Advance in time using 4th order Runge-Kutta Method
for i in range(nt):
    grid.rk4Step(dt)

# Plot grid
plt.figure("Solution Basis")
grid.plotSolution()
grid.plot()
plt.show()
