class Scheme:
    def __init__(self, points):
        raise NotImplementedError

    def getApproxFunction(self, values):
        raise NotImplementedError

    def getLeftCorrectionFunction(self):
        raise NotImplementedError

    def getRightCorrectionFunction(self):
        raise NotImplementedError

    def getLeftCorrectionFunctionGrad(self):
        raise NotImplementedError

    def getRightCorrectionFunctionGrad(self):
        raise NotImplementedError

    def plotBasisFunctions(self):
        raise NotImplementedError
