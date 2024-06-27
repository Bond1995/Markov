#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

api = wandb.Api()

# 
res1 = []
for run in api.runs("markov-PQ-est"):
    try:
        res1.append(
            (
                run.id,
                run.config["sequence_length"],
                run.summary["val/est_loss"],
                run.summary["val/baseline_est_loss"],
                run.summary["_step"],
            )
        )
    except:
        pass

#
df1 = pd.DataFrame(
    res1, columns=["id", "seq_length", "est_loss", "baseline_est_loss", "step"]
)

#%%
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.semilogx(df1.groupby("seq_length")["est_loss"].min(), color="tab:green", marker="o", label="Estimation loss")
ax.semilogx(df1.groupby("seq_length")["baseline_est_loss"].mean(), color="tab:red", marker="*", label="Baseline loss")
ax.grid(True, which="both")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
ax.legend(prop={'size': 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig("markov-est-loss.png")
