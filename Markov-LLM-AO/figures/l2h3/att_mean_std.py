import torch
import numpy as np
import matplotlib.pyplot as plt

n_layer = 2
for i in range(n_layer):
    att_mean = np.load('att_mean_'+str(i)+'.pt.npy')
    att_std = np.load('att_std_'+str(i)+'.pt.npy')
    for j in range(att_mean.shape[0]):
        fig, ax = plt.subplots()
        plt.imshow(att_mean[j,:32,:32], cmap='Purples', interpolation='nearest')
        #plt.colorbar(location='top', shrink=0.6)
        plt.colorbar()
        plt.savefig('att_mean_l'+str(i)+'h'+str(j)+'.pdf', bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.imshow(att_std[j,:32,:32], cmap='Purples', interpolation='nearest')
        #plt.colorbar(location='top', shrink=0.6)
        plt.colorbar()
        plt.savefig('att_std_'+str(i)+'h'+str(j)+'.pdf', bbox_inches='tight')
