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
for run in api.runs("linx/markov-simple-states-new", {"$and":[{"config.no_tying": False}, {"config.p": 0.9}]}):
    try:
        df1.append(run.history(samples=20000000))
    except:
        pass

df2 = []
losses2 = []
for run in api.runs("linx/markov-simple-states-new", {"$and":[{"config.no_tying": True}, {"config.p": 0.9}]}):
    try:
        df2.append(run.history(samples=20000000))
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

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(loss_mean1, color="tab:red", label="With weight-tying", linewidth=1)
ax.plot(loss_mean2, color="tab:purple", label="Without weight-tying", linewidth=1)
ax.fill_between(range(len(loss_mean1)), loss_mean1-loss_std1, loss_mean1+loss_std1, color="tab:red", alpha=0.2)
ax.fill_between(range(len(loss_mean2)), loss_mean2-loss_std2, loss_mean2+loss_std2, color="tab:purple", alpha=0.2)
ax.axhline(y = 1.601, color="black", linestyle = '--', label="Unigram loss", linewidth=2)
ax.axhline(y = 1.573, color="black", linestyle = 'dotted', label="Bigram loss", linewidth=2)
ax.set(xlabel="Iteration", ylabel="Test loss")
ax.xaxis.label.set_fontsize(16)
ax.yaxis.label.set_fontsize(16)
plt.xlim((0,4000))
#plt.ylim((0,2))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.legend(prop={'size': 16}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("loss09.pdf", bbox_inches='tight')
