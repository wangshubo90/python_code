import random

def bubble_sort(ls):
    '''
    bubble sort algorithm.

    '''
    n = len(ls)

    for i in range(n-1):
        for j in range(n-1-i):
            if ls[j] > ls[j+1]:
                ls[j+1], ls[j] = ls[j], ls[j+1]
            else:
                pass

def selection_sort(ls):
    n = len(ls)
    for i in range(n-1):
        smallest_idx = i
        for j in range(i+1,n):
            if ls[smallest_idx] > ls[j]:
                smallest_idx = j
            else:
                pass

        if smallest_idx !=i:
            ls[i], ls[smallest_idx] = ls[smallest_idx], ls[i]

def insert_sort(ls):
    n = len(ls)

    for i in range(1,n):
        for j in range(i):
            if ls[i] >= ls[j]:
                pass
            else:
                temp = ls[i]
                ls.pop(i)
                ls.insert(j,temp)

if __name__ == "__main__":
    a = random.sample(range(100),50)
    print(a)
    bubble_sort(a)
    print(a)