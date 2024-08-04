import torch
import numpy as np
import matplotlib.pyplot as plt

n_layer = 2
for i in range(n_layer):
    fig, ax = plt.subplots()
    att_mean = np.load('att_mean_'+str(i)+'.pt.npy')
    plt.imshow(att_mean[:32,:32], cmap='Purples', interpolation='nearest')
    #plt.colorbar(location='top', shrink=0.6)
    plt.colorbar()
    plt.savefig('att_mean_'+str(i)+'.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    att_std = np.load('att_std_'+str(i)+'.pt.npy')
    plt.imshow(att_std[:32,:32], cmap='Purples', interpolation='nearest')
    #plt.colorbar(location='top', shrink=0.6)
    plt.colorbar()
    plt.savefig('att_std_'+str(i)+'.pdf', bbox_inches='tight')
