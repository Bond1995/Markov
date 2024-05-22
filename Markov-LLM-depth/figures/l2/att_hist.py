import torch
import numpy as np
import matplotlib.pyplot as plt
 
fig, ax = plt.subplots()
att_mean = np.load('att_mean_0.pt.npy')
att_std = np.load('att_std_0.pt.npy')

data = att_mean[9,:10]
std = att_std[9,:10]
x = 0.5 + np.arange(data.shape[0])
plt.bar(x, data, width=1, color='green', edgecolor="white", linewidth=0.7, yerr=std, capsize=2)
ax.set(xlim=(0, data.shape[0]), ylim=(0, 1))
plt.savefig('att_hist.pdf', bbox_inches='tight')