import torch
import numpy as np
import matplotlib.pyplot as plt

wpe = []
c_fc = []
att_qkv = []
for i in [1, 100, 200]:
    fig, ax = plt.subplots()
    c_fc = np.load('files/c_fc-'+str(i)+'.pt.npy')
    plt.imshow(c_fc, cmap='Blues', interpolation='nearest')
    plt.colorbar(location='right')
    plt.savefig('c_fc-'+str(i)+'.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    att_qkv = np.load('files/att-qkv-'+str(i)+'.pt.npy')
    plt.imshow(att_qkv[16:], cmap='Blues', interpolation='nearest')
    plt.colorbar(location='top', shrink=0.6)
    plt.savefig('att_v-'+str(i)+'.pdf', bbox_inches='tight')
