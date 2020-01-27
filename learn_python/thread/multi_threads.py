#! /home/spl/ml/sitk/bin/python

# -*- coding: utf-8 -*-

import logging
import threading
import time
import concurrent.futures

def thread_function(name):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    
    '''
    j = 0
    while j < 10:
        j += 1
        threads = list()    
        for index in range(3):
            logging.info("Main    : create and start thread %d.", index)
            x = threading.Thread(target=thread_function, args=(index,))
            threads.append(x)
            x.start()
        
        for index, thread in enumerate(threads):
            logging.info("Main    : before joining thread %d.", index)
            thread.join()
            logging.info("Main    : thread %d done", index)
        
        print("==================================")
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        executor.map(thread_function,range(10))