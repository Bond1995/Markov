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
for run in api.runs("linx/markov-LLM-depth-heads", {"config.order": 1}):
    try:
        df1.append(run.history(samples=25000))
    except:
        pass

df2 = []
losses2 = []
for run in api.runs("linx/markov-LLM-depth-heads", {"config.order": 2}):
    try:
        df2.append(run.history(samples=25000))
    except:
        pass

df3 = []
losses3 = []
for run in api.runs("linx/markov-LLM-depth-heads", {"config.order": 3}):
    try:
        df3.append(run.history(samples=25000))
    except:
        pass

df4 = []
losses4 = []
for run in api.runs("linx/markov-LLM-depth-heads", {"config.order": 4}):
    try:
        df4.append(run.history(samples=25000))
    except:
        pass

#
for h in df1:
    opt_loss = np.nanmean(h["val/opt_loss"].values[:])
    loss = h["val/loss"].values[:] - opt_loss
    loss = moving_average(loss, n=10)
    losses1.append(loss)
    
loss1 = np.stack(losses1)
loss_mean1 = np.nanmean(loss1, axis=0)
loss_std1 = np.nanstd(loss1, axis=0)

for h in df2:
    opt_loss = np.nanmean(h["val/opt_loss"].values[:])
    loss = h["val/loss"].values[:] - opt_loss
    loss = moving_average(loss, n=10)
    losses2.append(loss)
    
loss2 = np.stack(losses2)
loss_mean2 = np.nanmean(loss2, axis=0)
loss_std2 = np.nanstd(loss2, axis=0)

for h in df3:
    opt_loss = np.nanmean(h["val/opt_loss"].values[:])
    loss = h["val/loss"].values[:] - opt_loss
    loss = moving_average(loss, n=10)
    losses3.append(loss)
    
loss3 = np.stack(losses3)
loss_mean3 = np.nanmean(loss3, axis=0)
loss_std3 = np.nanstd(loss3, axis=0)

for h in df4:
    opt_loss = np.nanmean(h["val/opt_loss"].values[:])
    loss = h["val/loss"].values[:] - opt_loss
    loss = moving_average(loss, n=10)
    losses4.append(loss)
    
loss4 = np.stack(losses4)
loss_mean4 = np.nanmean(loss4, axis=0)
loss_std4 = np.nanstd(loss4, axis=0)

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(loss_mean1, color="tab:blue", label=r"$k=1$", linewidth=2)
ax.plot(loss_mean2, color="tab:green", label=r"$k=2$", linewidth=2)
ax.plot(loss_mean3, color="tab:orange", label=r"$k=3$", linewidth=2)
ax.plot(loss_mean4, color="tab:purple", label=r"$k=4$", linewidth=2)
ax.fill_between(range(len(loss_mean1)), loss_mean1-loss_std1, loss_mean1+loss_std1, color="tab:blue", alpha=0.2)
ax.fill_between(range(len(loss_mean2)), loss_mean2-loss_std2, loss_mean2+loss_std2, color="tab:green", alpha=0.2)
ax.fill_between(range(len(loss_mean3)), loss_mean3-loss_std3, loss_mean3+loss_std3, color="tab:orange", alpha=0.2)
ax.fill_between(range(len(loss_mean4)), loss_mean4-loss_std4, loss_mean4+loss_std4, color="tab:purple", alpha=0.2)
ax.set(xlabel="Iteration", ylabel="Test loss gap from the optimal")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
plt.xlim((0,3000))
#plt.ylim((0.5,0.7))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(prop={'size': 12}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("loss3.pdf", bbox_inches='tight')