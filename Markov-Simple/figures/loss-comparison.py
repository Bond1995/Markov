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
for run in api.runs("linx/markov-comparison"):
    try:
        df1.append(run.history(samples=20000000))
    except:
        pass

df2 = []
losses2 = []
for run in api.runs("linx/markov-fixed-comparison"):
    try:
        df2.append(run.history(samples=20000000))
    except:
        pass

for h in df1:
    loss = h["val/loss"].values[:]
    loss = loss[~np.isnan(loss)]
    print(loss.shape)
    loss = moving_average(loss, n=10)
    losses1.append(loss)
    
loss1 = np.stack(losses1)
loss_mean1 = np.nanmean(loss1, axis=0)
loss_std1 = np.nanstd(loss1, axis=0)

for h in df2:
    loss = h["val/loss"].values[:]
    loss = loss[~np.isnan(loss)]
    loss=loss[:1000]
    loss = moving_average(loss, n=10)
    losses2.append(loss)
    
loss2 = np.stack(losses2)
loss_mean2 = np.nanmean(loss2, axis=0)
loss_std2 = np.nanstd(loss2, axis=0)

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(loss_mean1, color="tab:red", label="Full model with rank-one initialization", linewidth=1)
ax.plot(loss_mean2, color="tab:purple", label="Canonical model", linewidth=1)
ax.fill_between(range(len(loss_mean1)), loss_mean1-loss_std1, loss_mean1+loss_std1, color="tab:red", alpha=0.2)
ax.fill_between(range(len(loss_mean2)), loss_mean2-loss_std2, loss_mean2+loss_std2, color="tab:purple", alpha=0.2)
ax.axhline(y = 0.545, color="black", linestyle = 'dotted', label="Bigram loss", linewidth=2)
ax.set(xlabel="Iteration", ylabel="Test loss")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
plt.xlim((0,600))
#plt.ylim((0.60,0.9))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(prop={'size': 12}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("loss-comparison.pdf", bbox_inches='tight')
