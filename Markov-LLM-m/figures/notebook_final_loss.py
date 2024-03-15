#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

api = wandb.Api()

#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

api = wandb.Api()

#
run1 = api.run("linx/markov-LLM-1-final/vw76ir3z") # p + q < 1, without tying
df1 = run1.history()
loss1 = df1["val/loss"].values[:]
run2 = api.run("linx/markov-LLM-1-final/tf87x432") # p + q < 1, with tying
df2 = run2.history()
loss2 = df2["val/loss"].values[:]
#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(loss1, color="tab:green", label="                                  ", linewidth=3)
ax.plot(loss2, color="tab:purple", label="                              ", linewidth=3)
ax.axhline(y = 0.545, color="black", linestyle = '--', label="                 ", linewidth=3, zorder=1)
ax.set(xlabel=" ", ylabel=" ")
plt.locator_params(axis='y', nbins=6)
ax.xaxis.label.set_fontsize(18)
ax.yaxis.label.set_fontsize(18)
plt.xlim((0,100))
plt.ylim((0.5,0.7))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax.legend(prop={'size': 18}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("loss1.pdf")