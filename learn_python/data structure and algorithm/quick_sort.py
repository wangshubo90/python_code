def quick_sort(seq):
    N = len(seq)
    rec_quick_sort(seq, 0, N - 1)

def rec_quick_sort(seq, first, last):
    if first >= last:
        return
    else:
        pivot = seq[first]
        pos = pivot_partition(seq, first, last)

        rec_quick_sort(seq, first, pos - 1)
        rec_quick_sort(seq, pos + 1, last)

def pivot_partition(seq, first, last):
    pivot = seq[first]
    left = first + 1
    right = last

    while left <= right:
        while left <= right and seq[left] < pivot:
            left += 1
        
        while left <= right and seq[right] >= pivot:
            right -= 1

        if left < right:
            temp = seq[left]
            seq[left]  = seq[right]
            seq[right] = temp

    if right != first:
        seq[first] = seq[right]
        seq[right] = pivot

    return right