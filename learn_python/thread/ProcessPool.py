#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-
import logging
import threading
import time
import concurrent.futures
import math

PRIMES = [
    112272535467468456095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for idx,obj in enumerate(zip(PRIMES, executor.map(is_prime, PRIMES))):
            number,prime = obj
            print('%s: %d is prime: %s' % (idx,number, prime))

if __name__ == '__main__':
    main()