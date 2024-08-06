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
energy1 = []
for run in api.runs("linx/markov-simple-init-energy", {"config.init": "base"}):
    try:
        df1.append(run.history(samples=20000000))
    except:
        pass

df2 = []
energy2 = []
for run in api.runs("linx/markov-simple-init-energy", {"config.init": "lowrank"}):
    try:
        df2.append(run.history(samples=20000000))
    except:
        pass

for h in df1:
    energy = h["train/c_fc_energy"].values[:]
    energy = energy.astype(float)
    energy = energy[~np.isnan(energy)]
    #loss = moving_average(loss, n=10)
    energy1.append(energy)
    
energy1 = np.stack(energy1)
energy_mean1 = np.nanmean(energy1, axis=0)
energy_std1 = np.nanstd(energy1, axis=0)

for h in df2:
    energy = h["train/c_fc_energy"].values[:]
    energy = energy.astype(float)
    energy = energy[~np.isnan(energy)]
    #loss = moving_average(loss, n=10)
    energy2.append(energy)
    
energy2 = np.stack(energy2)
energy_mean2 = np.nanmean(energy2, axis=0)
energy_std2 = np.nanstd(energy2, axis=0)

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(energy_mean1, color="tab:red", label="Random initialization", linewidth=2)
ax.plot(energy_mean2, color="tab:purple", label="Rank-one initialization", linewidth=2)
ax.fill_between(range(len(energy_mean1)), energy_mean1-energy_std1, energy_mean1+energy_std1, color="tab:red", alpha=0.2)
ax.fill_between(range(len(energy_mean2)), energy_mean2-energy_std2, energy_mean2+energy_std2, color="tab:purple", alpha=0.2)
ax.set(xlabel="Iteration (multiples of 100)", ylabel="Fraction of energy")
ax.xaxis.label.set_fontsize(16)
ax.yaxis.label.set_fontsize(16)
plt.xlim((0,80))
plt.ylim((0,1.2))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.legend(prop={'size': 16}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("energy2.pdf", bbox_inches='tight')
