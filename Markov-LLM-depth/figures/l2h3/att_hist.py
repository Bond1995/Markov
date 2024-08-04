import torch
import numpy as np
import matplotlib.pyplot as plt
 
fig, ax = plt.subplots()
att = np.load('att_mean_0.pt.npy')
att_mean = att.mean(axis=0)
att_std = att.std(axis=0)

for j in range(att.shape[1]):
    data = att_mean[j,9,:10]
    std = att_std[j,9,:10]
    x = 0.5 + np.arange(data.shape[0])
    plt.figure()
    plt.bar(x, data, width=1, color='tab:purple', edgecolor="white", linewidth=0.7, yerr=std, capsize=2)
    ax.set(xlim=(0, data.shape[0]), ylim=(0, 1))
    plt.savefig('att_hist_l2h3_h'+str(j)+'.pdf', bbox_inches='tight')