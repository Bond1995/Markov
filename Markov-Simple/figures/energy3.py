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
energy2 = []
energy3 = []
energy4 = []
for run in api.runs("linx/markov-simple-init-energy-states"):
    try:
        df1.append(run.history(samples=20000000))
    except:
        pass

for h in df1:
    energy1t = h["train/att-v-energy1"].values[:]
    energy2t = h["train/att-v-energy2"].values[:]
    energy3t = h["train/att-v-energy3"].values[:]
    energy4t = h["train/att-v-energy4"].values[:]
    energy1t = energy1t.astype(float)
    energy2t = energy2t.astype(float)
    energy3t = energy3t.astype(float)
    energy4t = energy4t.astype(float)
    energy1t = energy1t[~np.isnan(energy1t)]
    energy2t = energy2t[~np.isnan(energy2t)]
    energy3t = energy3t[~np.isnan(energy3t)]
    energy4t = energy4t[~np.isnan(energy4t)]
    #loss = moving_average(loss, n=10)
    energy1.append(energy1t)
    energy2.append(energy2t)
    energy3.append(energy3t)
    energy4.append(energy4t)
    
energy1 = np.stack(energy1)
energy2 = np.stack(energy2)
energy3 = np.stack(energy3)
energy4 = np.stack(energy4)
energy_mean1 = np.nanmean(energy1, axis=0)
energy_mean2 = np.nanmean(energy2, axis=0)
energy_mean3 = np.nanmean(energy3, axis=0)
energy_mean4 = np.nanmean(energy4, axis=0)
energy_std1 = np.nanstd(energy1, axis=0)
energy_std2 = np.nanstd(energy2, axis=0)
energy_std3 = np.nanstd(energy3, axis=0)
energy_std4 = np.nanstd(energy4, axis=0)

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(energy_mean1, color="tab:red", label="Energy in first rank component", linewidth=2)
ax.plot(energy_mean2, color="tab:purple", label="Energy in first two rank components", linewidth=2)
ax.plot(energy_mean3, color="tab:orange", label="Energy in first three rank components", linewidth=2)
ax.plot(energy_mean4, color="tab:blue", label="Energy in first four rank components", linewidth=2)
ax.fill_between(range(len(energy_mean1)), energy_mean1-energy_std1, energy_mean1+energy_std1, color="tab:red", alpha=0.2)
ax.fill_between(range(len(energy_mean2)), energy_mean2-energy_std2, energy_mean2+energy_std2, color="tab:purple", alpha=0.2)
ax.fill_between(range(len(energy_mean3)), energy_mean3-energy_std3, energy_mean3+energy_std3, color="tab:orange", alpha=0.2)
ax.fill_between(range(len(energy_mean4)), energy_mean4-energy_std4, energy_mean4+energy_std4, color="tab:blue", alpha=0.2)
ax.set(xlabel="Iteration (multiples of 100)", ylabel="Fraction of energy")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
plt.xlim((0,100))
plt.ylim((0,1))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(prop={'size': 14}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("energy1-states.pdf", bbox_inches='tight')
