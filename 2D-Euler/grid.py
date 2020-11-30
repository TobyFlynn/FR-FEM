from element import Element
from edge import Edge, BoundaryEdge

class Grid2D:

    def __init__(self, bottomLeft, topRight, nx, ny):
        # Elements addressed as self.elements[x][y]
        self.elements = []
        self.edges = []
        self.boundaryEdges = []
        self.nx = nx
        self.dx = (topRight[0] - bottomLeft[0]) / self.nx
        self.ny = ny
        self.dy = (topRight[1] - bottomLeft[1]) / self.ny
        # Set boundary conditions
        self.leftBC = None
        self.rightBC = None
        self.bottomBC = None
        self.topBC = None
        # Order
        self.k = 3
        # Create grid
        for x in range(self.nx):
            yElem = []
            for y in range(self.ny):
                element = Element(x * self.dx, y * self.dy, self.k)
                yElem.append(element)
            self.elements.append(yElem)
        # Create internal edges (create edge to element to left and below)
        for x in range(1, self.nx - 1):
            for y in range(1, self.ny - 1):
                edge = Edge(self.elements[x - 1][y], self.elements[x][y])
                self.edges.append(edge)
                edge = Edge(self.elements[x][y - 1], self.elements[x][y])
                self.edges.append(edge)
        # Create edges for boundary elements
        # For y = 0
        for x in range(self.nx):
            if x == 0:
                edge = BoundaryEdge(self.leftBC, self.elements[x][0])
                self.boundaryEdges.append(edge)
            else:
                edge = Edge(self.elements[x - 1][0], self.elements[x][0])
                self.edges.append(edge)
            if x == self.nx - 1:
                edge = BoundaryEdge(self.elements[x][0], self.rightBC)
                self.boundaryEdges.append(edge)
            edge = BoundaryEdge(self.bottomBC, self.elements[x][0])
            self.boundaryEdges.append(edge)
        # For y = ny - 1
        for x in range(self.nx):
            if x == 0:
                edge = BoundaryEdge(self.leftBC, self.elements[x][self.ny - 1])
                self.boundaryEdges.append(edge)
            else:
                edge = Edge(self.elements[x - 1][self.ny - 1], self.elements[x][self.ny - 1])
                self.edges.append(edge)
            if x == self.nx - 1:
                edge = BoundaryEdge(self.elements[x][self.ny - 1], self.rightBC)
                self.boundaryEdges.append(edge)
            edge = BoundaryEdge(self.elements[x][self.ny - 1], self.topBC)
            self.boundaryEdges.append(edge)
            edge = Edge(self.elements[x][self.ny - 2], self.elements[x][self.ny - 1])
            self.edges.append(edge)
        # For x = 0
        for y in range(1, self.ny - 1):
            edge = BoundaryEdge(self.leftBC, self.elements[0][y])
            self.boundaryEdges.append(edge)
            edge = Edge(self.elements[0][y - 1], self.elements[0][y])
            self.edges.append(edge)
        # For x = nx - 1
        for y in range(1, self.ny - 1):
            edge = BoundaryEdge(self.elements[self.nx - 1][y], self.rightBC)
            self.boundaryEdges.append(edge)
            edge = Edge(self.elements[self.nx - 1][y - 1], self.elements[self.nx - 1][y])
            self.edges.append(edge)
            edge = Edge(self.elements[self.nx - 2][y], self.elements[self.nx - 1][y])
            self.edges.append(edge)
