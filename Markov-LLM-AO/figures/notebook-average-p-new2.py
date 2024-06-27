#%%
import wandb
import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
api = wandb.Api()


# %% Markov LLM final loss (6k iterations)
res = []
for run in api.runs("markov-PQ-add-beta"):
    try:
        data = run.history()
        c = data["batch1-q"].values[:]
        res.append((run.config.get("p", None), run.config.get("q", None), np.average(c[~(np.isnan(c))])))
    except:
        pass

df = pd.DataFrame(res, columns=["p", "q", "avg"])
piv = df.pivot_table(index="p", columns="q", values="avg")
fig, ax = plt.subplots(figsize=(10, 10))
v = ax.matshow(piv, vmin=0.2, vmax=0.2, cmap='Greys')
x=np.arange(len(piv.index)+1)-0.5
y=4-x
stepf = ax.step(x,y,where='post', color='black', linewidth=2)
reg1 = ax.fill_between(x, y, -0.5, step='post', color='blue', alpha=0.2)
reg2 = ax.fill_between(x, y, 4.5, step='post', color='red', alpha=0.2)
ax.set_xticks(range(len(piv.columns)), piv.columns)
ax.set_xticks(np.arange(len(piv.columns))+0.5, minor=True)
ax.set_yticks(range(len(piv.index)), piv.index)
ax.set_yticks(np.arange(len(piv.index))+0.5, minor=True)
ax.grid(which='minor')
ax.xaxis.set_ticks_position("bottom")
#ax.set(xlabel=r"$p$", ylabel=r"$q$")
#ax.set_title(r"Predicted probability $f_{\theta}(x^n | x_n = 0)$ (average across 5 runs)")
#fig.colorbar(v, ax=ax, label=r"Average predicted $q$")
for i in range(len(piv.index)):
    for j in range(len(piv.columns)):
        ax.text(j, i, f"{piv.iloc[i, j]:.3f}", ha="center", va="center", color="black", fontsize=16)
ax.text(2.48,0.7,r"$\approx p$", color="red", fontsize=18)
rec = patches.Rectangle((1.55,-0.45), 0.9, 2.9, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rec)
ax.legend([reg1, reg2, stepf[0]], ["                       ", "       ", "      "], loc=(1.01,0.86), prop={'size': 16})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig("markov-avg-p.pdf", bbox_inches='tight')

'''# %% PowerSGD varying rank (very high power)
res = []
for run in api.runs("federated-codes", {"group": "psgdstudy"}):
    try:
        res.append((run.id, run.config.get("powersgd_rank", None), run.config["learning_rate"], run.summary["best_accuracy"], run.summary["_step"]))
    except:
        pass

df = pd.DataFrame(res, columns=["id", "powersgd_rank", "learning_rate", "best_accuracy", "step"])
piv = df.pivot_table(index="powersgd_rank", columns="learning_rate", values="best_accuracy")
fig, ax = plt.subplots(figsize=(8, 6))
v = ax.matshow(piv, vmin=0.6, vmax=0.85)
ax.set_xticks(range(len(piv.columns)), piv.columns)
ax.set_yticks(range(len(piv.index)), piv.index.astype(int))
ax.xaxis.set_ticks_position("bottom")
ax.set(xlabel="Learning rate", ylabel="PowerSGD Rank")
ax.set_title("Effect of PowerSGD rank (100 epochs, Cifar-10, ResNet-18)")
fig.colorbar(v, ax=ax, label="Final accuracy (exponential moving average)")
for i in range(len(piv.index)):
    for j in range(len(piv.columns)):
        ax.text(j, i, f"{100*piv.iloc[i, j]:.0f}%", ha="center", va="center", color="w")
fig.savefig("powersgd_rankstudy.png", dpi=300)
# %%'''

# p = torch.Tensor([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
# H = (-p * torch.log(p) - (1-p)*torch.log(1-p)) / (p+1)
# print(H)
# print(df.loc[df['q']==1]['val/loss'].values[0])