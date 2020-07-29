# compute n!
def fact(n):
    assert n >= 0, "Fractorial not defined for negative integer"
    if n < 2:
        return 1
    else:
        return return fact(n-1)*n

def exponential(value , power):
    assert power >= 0, "Only accept positive power"
    if power == 0:
        return 1
    elif power == 1:
        return value
    elif power >1:
        return exponential(value*value, power//2) * exponential(value, power % 2)

def exponential(value, power):
    if power == 0:
        return 1
    if power % 2 == 0 :
        return exponential(value * value, power // 2)
    else:
        return exponential(value * value, power // 2) * value

def binary search(value,seq, first, last):
    if first > last:
        return False
    else:
        mid = (first + last) // 2

        if seq[mid] == value:
            return mid
        elif seq[mid] < value:
            return binary_search(value, seq, mid + 1, last )
        else seq[mid] > value:
            return binary_search(value, seq, first, mid -1)