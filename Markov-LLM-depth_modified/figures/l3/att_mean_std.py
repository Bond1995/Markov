import torch
import numpy as np
import matplotlib.pyplot as plt


for i in range(3):
    fig, ax = plt.subplots()
    att_mean = np.load('att_mean_'+str(i)+'.pt.npy')
    plt.imshow(att_mean, cmap='Greens', interpolation='nearest')
    #plt.colorbar(location='top', shrink=0.6)
    plt.colorbar()
    plt.savefig('att_mean_'+str(i)+'.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    att_std = np.load('att_std_'+str(i)+'.pt.npy')
    plt.imshow(att_std, cmap='Greens', interpolation='nearest')
    #plt.colorbar(location='top', shrink=0.6)
    plt.colorbar()
    plt.savefig('att_std_'+str(i)+'.pdf', bbox_inches='tight')
