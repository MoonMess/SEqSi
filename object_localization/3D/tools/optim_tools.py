import numpy as np


def iterative_grid_search_min(fun, bounds, nb_splits=10, nb_iter=3, verbose = False):
    splits = np.arange(0,nb_splits+1)/nb_splits
    start, stop = bounds 
    for _ in range(nb_iter):
        crt_splits = start + splits*(stop-start)
        values = [fun(x) for x in crt_splits]
        i = np.argmin(values)
        if verbose: print(f"Crt x: {round(crt_splits[i], 3)}, crt fx: {values[i]}")
        dx = crt_splits[1]-crt_splits[0]
        start, stop = crt_splits[i] - dx, crt_splits[i] + dx
    
    return crt_splits[i]
