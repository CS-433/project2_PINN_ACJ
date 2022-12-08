import csv
import numpy as np
import pandas as pd

def data_processing(data_path):
    data = np.asarray(pd.read_csv(data_path))
    nb_var = len(data[0,:])
    std_var = np.std(data, axis=0)
    gaussian_noise = np.zeros([data.shape[0], data.shape[1]])
    mu = 0

    for j in range(nb_var):
       if j==0:
        sigma = 0
        gaussian_noise[:,j] = np.random.normal(mu, sigma, [data.shape[0]])
       else:
        sigma = 0.003*std_var[j]
        gaussian_noise[:,j] = np.random.normal(mu, sigma, [data.shape[0]])
    #data[:,j+1] = data[:,j+1] + gaussian_noise

    data_process = data+gaussian_noise
    return data_process




    
