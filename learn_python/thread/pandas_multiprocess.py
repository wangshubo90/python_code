import pandas as pd 
import numpy as np 
import os 
import concurrent.futures
'''
def multi_process_df(function, dataframe, workers = 10, **kwds):
    """
    Description:
    Arg(s):
        function: a function to process dataframe
        dataframe: pd.DataFrame, to be processed parallelly 
        workers: int, number of processes
        **kwds: other arguments to be passes to the function
    Return(s):
        df: pd.DataFrame, after concatenation 
    """
    row_number = dataframe.shape[0]
    batch_size = row_number // workers 
    dataframe_batches = []    
    for i in range(workers-1):
        dataframe_batches.append(dataframe[batch_size*i:batch_size*(i+1)])
    
    dataframe_batches.append(dataframe[batch_size*(workers - 1):])

    results_ls = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for _, result in enumerate(executor.map(function,dataframe_batches)):
            results_ls.append(result)
    
    return results_ls
'''