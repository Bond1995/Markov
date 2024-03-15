#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

api = wandb.Api()

# 
res1 = []
for run in api.runs("markov-LLM-2-final-memory-800", {"config.n_embd": 32}):
    try:
        res1.append(
            (
                run.id,
                run.config["memory"],
                run.summary["val/loss"],
                run.summary["_step"],
            )
        )
    except:
        pass

#
df1 = pd.DataFrame(
    res1, columns=["id", "memory", "loss", "step"]
)
# 
res2 = []
for run in api.runs("markov-LLM-2-final-memory-layers", {"config.n_layer": 4}):
    try:
        res2.append(
            (
                run.id,
                run.config["memory"],
                run.summary["val/loss"],
                run.summary["_step"],
            )
        )
    except:
        pass

#
df2 = pd.DataFrame(
    res2, columns=["id", "memory", "loss", "step"]
)
# 
res3 = []
for run in api.runs("markov-LLM-2-final-memory-layers", {"config.n_layer": 8}):
    try:
        res3.append(
            (
                run.id,
                run.config["memory"],
                run.summary["val/loss"],
                run.summary["_step"],
            )
        )
    except:
        pass

#
df3 = pd.DataFrame(
    res3, columns=["id", "memory", "loss", "step"]
)

#%%
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(df1.groupby("memory")["loss"].mean(), color="tab:green", marker="o", label="               ")
ax.plot(df2.groupby("memory")["loss"].mean(), color="tab:orange", marker="o", label="               ")
ax.plot(df3.groupby("memory")["loss"].mean(), color="tab:purple", marker="o", label="               ")
ax.axhline(y = 0.545, color="black", linestyle = '--', label="                 ", linewidth=2, zorder=1)
ax.axhline(y = 0.673, color="black", linestyle = ':', label="                 ", linewidth=2, zorder=1)
ax.grid(True, which="both")
plt.locator_params(axis='y', nbins=6)
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
ax.legend(loc=(0.07,0.58), prop={'size': 12}, handlelength=1.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig("memory-loss.pdf")
