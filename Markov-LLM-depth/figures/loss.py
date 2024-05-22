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

df2 = []
losses2 = []
for run in api.runs("linx/markov-LLM-order-depth-final-2", {"config.order": 2}):
    try:
        df2.append(run.history(samples=25000))
    except:
        pass

df4 = []
losses4 = []
for run in api.runs("linx/markov-LLM-order-depth-final-2", {"config.order": 4}):
    try:
        df4.append(run.history(samples=25000))
    except:
        pass

df8 = []
losses8 = []
for run in api.runs("linx/markov-LLM-order-depth-final-2", {"config.order": 8}):
    try:
        df8.append(run.history(samples=25000))
    except:
        pass

#
for h in df2:
    opt_loss = np.nanmean(h["val/opt_loss"].values[:])
    loss = h["val/loss"].values[:] - opt_loss
    loss = moving_average(loss, n=50)
    losses2.append(loss)
    
loss2 = np.stack(losses2)
loss_mean2 = np.nanmean(loss2, axis=0)
loss_std2 = np.nanstd(loss2, axis=0)

for h in df4:
    opt_loss = np.nanmean(h["val/opt_loss"].values[:])
    loss = h["val/loss"].values[:] - opt_loss
    loss = moving_average(loss, n=50)
    losses4.append(loss)
    
loss4 = np.stack(losses4)
loss_mean4 = np.nanmean(loss4, axis=0)
loss_std4 = np.nanstd(loss4, axis=0)

for h in df8:
    opt_loss = np.nanmean(h["val/opt_loss"].values[:])
    loss = h["val/loss"].values[:] - opt_loss
    loss = moving_average(loss, n=50)
    losses8.append(loss)
    
loss8 = np.stack(losses8)
loss_mean8 = np.nanmean(loss8, axis=0)
loss_std8 = np.nanstd(loss8, axis=0)

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(loss_mean2[:-2], color="tab:blue", label="          ", linewidth=2)
ax.plot(loss_mean4[:-2], color="tab:green", label=" ", linewidth=2)
ax.plot(loss_mean8[:-2], color="tab:orange", label=" ", linewidth=2)
ax.fill_between(range(len(loss_mean2)), loss_mean2-loss_std2, loss_mean2+loss_std2, color="tab:blue", alpha=0.2)
ax.fill_between(range(len(loss_mean4)), loss_mean4-loss_std4, loss_mean4+loss_std4, color="tab:green", alpha=0.2)
ax.fill_between(range(len(loss_mean8)), loss_mean8-loss_std8, loss_mean8+loss_std8, color="tab:orange", alpha=0.2)
#ax.set(xlabel="Iterations", ylabel="Test loss")
ax.xaxis.label.set_fontsize(18)
ax.yaxis.label.set_fontsize(18)
plt.xlim((0,25000))
#plt.ylim((0.5,0.7))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax.legend(prop={'size': 18}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("loss.pdf", bbox_inches='tight')