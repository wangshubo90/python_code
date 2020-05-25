from Sorting import merge_ordered

def merge_sort(ls):
    l = len(ls)
    
    if l == 1:
        return ls

    else:
        mid = l // 2
        left = merge_sort(ls[0:mid])
        right = merge_sort(ls[mid:])

        return merge_ordered(left, right)

from oned_array import Array

def vir_merge_sort(seq):
    N = len(seq)
    tem_seq = Array(N)
    first = 0
    last = N - 1

    def merge_ordered(seq, first, mid, end, tem_seq):
        a = first
        m = mid
        i = 0
        while a < mid and m < end:
            if seq[a] < seq[m]:
                tem_seq[i] = seq[a]
                a += 1
                i += 1
            else:
                tem_seq[i] = seq[m]
                i += 1
                m += 1

        while a < mid:
            tem_seq[i] = seq[a]
            a += 1
            i += 1
        while m < end:
            tem_seq[i] = seq[m]
            m += 1
            i += 1

        for j in range(end - first):
            seq[j + first] = tem_seq[j]

    def merge_sort(seq, first, last, tem_seq):
        if first == last :
            return
        else:
            mid = (first + last) // 2
            merge_sort(seq, first, mid, tem_seq)
            merge_sort(seq,  mid+1, last, tem_seq)

            merge_ordered(seq, first, mid + 1, last + 1, tem_seq)

    merge_sort(seq, first, last, tem_seq)

"""
Time complexity of merge_sort algorithm:
    len = N
    first level:  N/2 * 2
    second level: N/4 * 4
    ...
    Total log N levels
    sumup : N * log N
"""

def quick_sort(seq):
    if len(seq) 
    if len(seq) == 2:
        if seq[0] <= seq[1]:
            pass
        else:
            seq[0], seq[1] = seq[1], seq[0]

    else:
