#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

api = wandb.Api()

# 
res1 = []
for run in api.runs("markov-PQ-seq-length-3"):
    try:
        res1.append(
            (
                run.id,
                run.config["sequence_length"],
                run.summary["val/pred_loss"],
                run.summary["val/baseline_loss"],
                run.summary["_step"],
            )
        )
    except:
        pass

#
df1 = pd.DataFrame(
    res1, columns=["id", "seq_length", "pred_loss", "baseline_loss", "step"]
)

#%%
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.semilogx(df1.groupby("seq_length")["pred_loss"].mean(), color="tab:green", marker="o", label="Prediction loss")
ax.semilogx(df1.groupby("seq_length")["baseline_loss"].mean(), color="tab:red", marker="*", label="Baseline loss (corrected)")
ax.grid(True, which="both")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
ax.legend(prop={'size': 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig("markov-pred-loss3.png")
