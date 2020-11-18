from element import Element

# Number of elements
nx = 40
# Size of element
dx = 0.1
# Time step
dt = 1
# Number of solution points in an element
k = 4

firstElement = Element(k)

# prevElement = firstElement
#
# # Construct 1D regular mesh of elements
# for x in range(1, nx):
#     newElement = Element(k)
#     newElement.setRightElement(prevElement)
#     prevElement = newElement
