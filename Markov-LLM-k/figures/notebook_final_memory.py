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

#%%
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(df1.groupby("memory")["loss"].mean(), color="tab:green", marker="o", label="               ")
ax.grid(True, which="both")
plt.locator_params(axis='y', nbins=6)
ax.xaxis.label.set_fontsize(15)
ax.yaxis.label.set_fontsize(15)
#ax.legend(prop={'size': 15})
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.savefig("memory-loss.pdf")
