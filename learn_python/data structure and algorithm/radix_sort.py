from Queue import Queue_lk as Queue
from oned_array import Array

def radix_sort(intList, numDigit):
    binArray = Array(10)
    for i in range(10):
        binArray[i] = Queue()

    level = 1

    for d in range(numDigit):
        for key in intList:
            digit = (key // level) % 10
            binArray[digit].enQueue(key)
        i = 0
        for bin in binArray:
            while not bin._isEmpty() :
                intList[i] = bin.deQueue()
                i += 1

        level *= 10

