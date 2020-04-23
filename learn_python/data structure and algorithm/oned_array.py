# Implements the Array ADT using array capabilities of the ctypes module. 2 import ctypes 3 4 class Array : 5
# Creates an array with size elements.
import ctypes

class Array:
    def __init__( self, size ):
        assert size > 0, "Array size must be > 0"
        self._size = size
        # Create the array structure using the ctypes module.
        PyArrayType = ctypes.py_object * size
        self._elements = PyArrayType()
        # Initialize each element.
        self.clear( None )
        # Returns the size of the array.
    def __len__( self ):
        return self._size
    # Gets the contents of the index element.
    def __getitem__( self, index ):
        assert index >= 0 and index < len(self), "Array subscript out of range"
        return self._elements[ index ]

    # Puts the value in the array element at index position.
    def __setitem__( self, index, value ):
        assert index >= 0 and index < len(self), "Array subscript out of range"
        self._elements[ index ] = value

    # Clears the array by setting each element to the given value.
    def clear( self, value ):
        for i in range( len(self) ) :
            self._elements[i] = value

    # Returns the array's iterator for traversing the elements.
    def __iter__( self ):
        return _ArrayIterator( self._elements )
    # An iterator for the Array ADT. 

class _ArrayIterator :
    def __init__( self, theArray ): 
        self._arrayRef = theArray
        self._curNdx = 0
    def __iter__( self ): 
        return self
    def __next__( self ):
        if self._curNdx < len( self._arrayRef ) : 
            entry = self._arrayRef[ self._curNdx ]
            self._curNdx += 1 
            return entry
        else :
            raise StopIteration

class Array2D:
    def __init__( self, numRows, numCols ):
        self._theRows = Array( numRows )
        for i in range( numRows ) :
            self._theRows[i] = Array( numCols )
    
    def numRows( self ):
        return len( self._theRows )
    
    def numCols( self ):
        return len( self._theRows[0] )
    
    def clear( self, value ):
        for row in range( self.numRows() ):
            self._theRows[row].clear( value )
    
    def __getitem__( self, ndxTuple ):
        assert len(ndxTuple) == 2, "Invalid number of array subscripts."
        row = ndxTuple[0] 
        col = ndxTuple[1]
        assert row >= 0 and row < self.numRows() \
             and col >= 0 and col < self.numCols(), \
                  "Array subscript out of range."
        the1dArray = self._theRows[row] 
        return the1dArray[col]
    
    def __setitem__( self, ndxTuple, value ):
        assert len(ndxTuple) == 2, "Invalid number of array subscripts." 
        row = ndxTuple[0] 
        col = ndxTuple[1]
        assert row >= 0 and row < self.numRows() \
            and col >= 0 and col < self.numCols(), \
                "Array subscript out of range."
        the1dArray = self._theRows[row] 
        the1dArray[col] = value

class MultiArray:
    def __init__(self, *dimensions):
        assert len(dimensions) >1, "The array must have 2 or more dimensions."
        self._dims = dimensions
        size = 1
        for d in dimensions :
            assert d > 0, "Dimensions must be > 0"
            size *= d
        
        self._elements = Array(size)
        self._factors = Array(len(dimensions))
        self._computeFactors()

    def numDims(self):
        return len(self._dims)

    def len(self,dim):
        assert dim >= 1 and dim <= len(self._dims), "Dimension component out of range"
        return self._dims[dim-1]

    def clear(self, value):
        self._elements.clear(value)
    
    def __getitem__(self, ndxTuple):
        assert len(ndxTuple) == len(self._dims), "Invalid number of array subsripts"
        index = self._computeIndex( ndxTuple)
        assert index is not None, "Array subscript out of range"
        return self._elements[index]

    def __setitem__(self, ndxTuple, value):
        assert len(ndxTuple) == len(self._dims), "Invalid number of array subsripts"
        index = self._computeIndex( ndxTuple)
        assert index is not None, "Array subscript out of range"
        self._elements[index] = value
    
    def _computeIndex(self, idx):
        offset = 0
        for j in range(len(idx)):
            if idx[j] < 0 and idx[j] >= self._dims[j]:
                return None
            else:
                offset += idx[j] * self._factors[j]
        return offset

    def _computeFactors(self):
        self._factors[self.numDims()-1] = 1
        for i in range(self.numDims()-2,-1,-1):
            self._factors[i] = self._factors[i+1]*self._dims[i+1]


    