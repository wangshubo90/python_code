from oned_array import Array


class SparseMatrix:
    # create a sparse matrix of size of numRows x numCols initialized to 0
    def __init__(self, numRows, numCols):
        self._listofRows = Array(numRows)
        self._numCols = numCols

    def numRows(self):
        return len(self._listofRows)

    def numCols(self):
        return self._numCols

    def __getitem__(self, ndxTuple):
        row, col = ndxTuple
        curNode = self._listofRows[row]
        while curNode is not None and curNode.col != col:
            curNode = curNode.next
        
        if curNode is not None:
            return curNode.value
        else:
            return 0.0

    def __setitem__(self, ndxTuple, value):
        row, col = ndxTuple        
        predNode = None
        curNode = self._listofRows[row]
        while curNode is not None and curNode.col != col:
            predNode = curNode
            curNode = curNode.next

        if curNode is not None and curNode.col == col: # check whether the new node in already in the matrix
            if value == 0.0: # if value == 0.0, we remove the node
                if curNode == self._listofRows[row]: # check if curNode is the head of [row]
                    self._listofRows[row] == curNode.next # move the head to the next row
                else:
                    predNode.next = curNode.next
            else:
                curNode.value = value

        elif value != 0.0: # new node is not in the matrix yet.
            newNode = _MatrixElementNode(col,value)
            newNode.next = curNode
            if curNode ==self._listofRows[row]:
                self._listofRows[row] = newNode
            else:
                predNode.next = newNode

    def scaleBy(self, scalar):
        for row in range(self.numRows()):
            curNode = self._listofRows[row]
            while curNode is not None:
                curNode.value *= scalar
                curNode = curNode.next

    def __add__(self, rhsMatrix):
        assert rhsMatrix.numCols() == self.numCols() and rhsMatrix.numRows() == self.numCols(),\
                    "Matrix sizes are not compatable for adding."

        newMatrix = SparseMatrix(self.numRows(),self.numCols())
        for row in range(self.numRows()):
            curNode = self._listofRows[row]
            while curNode is not None:
                newMatrix[row, curNode.col] = curNode.value
                curNode = curNode.next
        
        for row in range(rhsMatrix.numRows()):
            curNode = rhsMatrix._listofRows[row]
            while curNode is not None:
                value = newMatrix[row, curNode.col]
                value += curNode.value
                newMatrix[row, curNode.col] = value
                curNode = curNode.next

        return newMatrix

class _MatrixElementNode :
    def __init__( self, col, value ) :
        self.col = col
        self.value = value
        self.next = None

