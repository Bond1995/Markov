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

run1 = api.run("linx/markov-simple-init-good/25vu3t7d")
h1 = run1.history(samples=200000000)
loss1 = h1["val/loss"].values[:]
loss1 = loss1[~np.isnan(loss1)]
loss1 = moving_average(loss1, n=10)

run2 = api.run("linx/markov-simple-init-good/z3iz9z8k")
h2 = run2.history(samples=200000000)
loss2 = h2["val/loss"].values[:]
loss2 = loss2[~np.isnan(loss2)]
loss2 = moving_average(loss2, n=10)

def moving_average(a, n=3):
	t = np.floor(n/2).astype(int)
	b = np.zeros(a.shape)
	for i in range(b.shape[-1]):
		b[i] = np.mean(a[max(0, i-t):min(i+t+1, a.shape[-1])])
	
	return b

'''df1 = []
losses1 = []
for run in api.runs("linx/markov-simple-init-good", {"config.init": "base"}):
    try:
        df1.append(run.history(samples=20000))
    except:
        pass

df2 = []
losses2 = []
for run in api.runs("linx/markov-simple-init-good", {"config.init": "good"}):
    try:
        df2.append(run.history(samples=20000))
    except:
        pass

#
for h in df1:
    loss = h["val/loss"].values[:1000]
    #loss = moving_average(loss, n=10)
    losses1.append(loss)
    
loss1 = np.stack(losses1)
loss_mean1 = np.nanmean(loss1, axis=0)
loss_std1 = np.nanstd(loss1, axis=0)

for h in df2:
    loss = h["val/loss"].values[:1000]
    #loss = moving_average(loss, n=10)
    losses2.append(loss)
    
loss2 = np.stack(losses2)
loss_mean2 = np.nanmean(loss2, axis=0)
loss_std2 = np.nanstd(loss2, axis=0)'''

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(range(450), loss1[:450], color="tab:red", label="                                      ", linewidth=2)
ax.plot(range(450), loss2[:450], color="purple", label=" ", linewidth=2)
#ax.fill_between(range(len(loss_mean1)), loss_mean1-loss_std1, loss_mean2+loss_std2, color="tab:red", alpha=0.2)
#ax.fill_between(range(len(loss_mean2)), loss_mean2-loss_std2, loss_mean2+loss_std2, color="tab:blue", alpha=0.2)
ax.axhline(y = 0.665, color="black", linestyle = '--', label=" ", linewidth=2)
ax.axhline(y = 0.63, color="black", linestyle = 'dotted', label=" ", linewidth=2)
#ax.set(xlabel="Iteration", ylabel="Test loss")
ax.xaxis.label.set_fontsize(16)
ax.yaxis.label.set_fontsize(16)
plt.xlim((0,450))
plt.ylim((0.60,0.9))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.legend(prop={'size': 16}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("loss.pdf", bbox_inches='tight')