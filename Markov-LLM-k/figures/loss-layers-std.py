#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

api = wandb.Api()

def moving_average(a, n=3):
	t = np.floor(n/2).astype(int)
	b = np.zeros(a.shape)
	for i in range(b.shape[-1]):
		b[i] = np.mean(a[max(0, i-t):min(i+t+1, a.shape[-1])])
	
	return b

df1 = []
losses1 = []
for run in api.runs("mamba-markov/markov-LLM-layers-iclr", {"config.n_layer": 1}):
    try:
        df1.append(run.history(samples=20000000))
    except:
        pass

df2 = []
losses2 = []
for run in api.runs("mamba-markov/markov-LLM-layers-iclr", {"config.n_layer": 2}):
    try:
        df2.append(run.history(samples=20000000))
    except:
        pass

df4 = []
losses4 = []
for run in api.runs("mamba-markov/markov-LLM-layers-iclr", {"config.n_layer": 4}):
    try:
        df4.append(run.history(samples=20000000))
    except:
        pass

df8 = []
losses8 = []
for run in api.runs("mamba-markov/markov-LLM-layers-iclr", {"config.n_layer": 8}):
    try:
        df8.append(run.history(samples=20000000))
    except:
        pass

for h in df1:
    loss = h["val/loss"].values[:]
    loss = loss[~np.isnan(loss)]
    loss = moving_average(loss, n=10)
    losses1.append(loss)
    
loss1 = np.stack(losses1)
loss_mean1 = np.nanmean(loss1, axis=0)
loss_std1 = np.nanstd(loss1, axis=0)

for h in df2:
    loss = h["val/loss"].values[:]
    loss = loss[~np.isnan(loss)]
    loss = moving_average(loss, n=10)
    losses2.append(loss)
    
loss2 = np.stack(losses2)
loss_mean2 = np.nanmean(loss2, axis=0)
loss_std2 = np.nanstd(loss2, axis=0)

for h in df4:
    loss = h["val/loss"].values[:]
    loss = loss[~np.isnan(loss)]
    loss = moving_average(loss, n=10)
    losses4.append(loss)
    
loss4 = np.stack(losses4)
loss_mean4 = np.nanmean(loss4, axis=0)
loss_std4 = np.nanstd(loss4, axis=0)

for h in df8:
    loss = h["val/loss"].values[:]
    loss = loss[~np.isnan(loss)]
    loss = moving_average(loss, n=10)
    losses8.append(loss)
    
loss8 = np.stack(losses8)
loss_mean8 = np.nanmean(loss8, axis=0)
loss_std8 = np.nanstd(loss8, axis=0)

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(loss_mean1, color="tab:red", label="                                            ", linewidth=2)
ax.plot(loss_mean2, color="tab:green", label=" ", linewidth=2)
ax.plot(loss_mean4, color="tab:orange", label=" ", linewidth=2)
ax.plot(loss_mean8, color="tab:purple", label=" ", linewidth=2)
ax.fill_between(range(len(loss_mean1)), loss_mean1-loss_std1, loss_mean1+loss_std1, color="tab:red", alpha=0.2)
ax.fill_between(range(len(loss_mean2)), loss_mean2-loss_std2, loss_mean2+loss_std2, color="tab:green", alpha=0.2)
ax.fill_between(range(len(loss_mean4)), loss_mean4-loss_std4, loss_mean4+loss_std4, color="tab:orange", alpha=0.2)
ax.fill_between(range(len(loss_mean8)), loss_mean8-loss_std8, loss_mean8+loss_std8, color="tab:purple", alpha=0.2)
ax.axhline(y = 0.666, color="black", linestyle = 'dotted', label=" ", linewidth=2)
ax.axhline(y = 0.619, color="black", linestyle = '--', label=" ", linewidth=2)
#ax.set(xlabel=" ", ylabel=" ")
#ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
plt.xlim((0,500))
plt.ylim((0.6,0.8))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#ax.legend(prop={'size': 12}, handlelength=1.7)
ax.legend(prop={'size': 15})
ax.grid(True, which="both")
fig.set_size_inches(9,5)
fig.savefig("loss-layers-std.pdf", bbox_inches='tight')
