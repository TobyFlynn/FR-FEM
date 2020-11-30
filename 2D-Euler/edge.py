

class BoundaryEdge:

    def __init__(self, boundary, element):
        self.boundary = boundary
        self.element = element

class Edge:

    def __init__(self, element1, element2):
        self.element1 = element1
        self.element2 = element2
