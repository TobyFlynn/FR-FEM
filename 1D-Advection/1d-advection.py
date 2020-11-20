from grid import Grid
from element import Element

import numpy as np
import matplotlib.pyplot as plt

e = Element(4, 1, 1)
e.plotBasisFunctions()
e.plotCorrectionFunctions()
plt.show()

# Number of elements
nx = 10
# Number of time steps
nt = 1
# Number of solution points in an element
k = 4
# Wave speed
a = 1

# Create 1D regular grid and set initial condition
grid = Grid((0.0, 1.0), nx, k, lambda x: np.exp(-40 * (x - 0.5)**2))
# CFL number (0.9 * CFL limit)
CFL = 0.9 * 0.145
# dt
dt = (CFL * grid.getdx()) / abs(a)
print("CFL: " + str(CFL))
print("dt: " + str(dt))

for i in range(nt):
    grid.rk4Step(dt)
    # grid.eulerStep(dt)

# Plot grid
plt.figure("Solution")
grid.plotSolution()
plt.show()
