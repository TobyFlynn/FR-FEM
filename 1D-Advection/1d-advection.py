from grid import Grid

import numpy as np

# Number of elements
nx = 10
# Size of element
dx = 0.1
# Number of time steps
nt = 100
# Time step
dt = 1
# Number of solution points in an element
k = 8

# Create 1D regular grid and set initial condition
grid = Grid((0.0, 1.0), nx, k, lambda x: np.exp(-40 * (x - 0.5)**2))

# Plot grid
grid.plot()
