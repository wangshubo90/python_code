def binary_search(ls,value):
    ls.sort()
    n = len(ls)
    low = 0
    high = n-1

    while low <= high:
        mid = (low + high) // 2

        if value < ls[mid]:
            high = mid - 1
        elif value > ls[mid]:
            low = mid + 1
        else:
            return mid

    return None