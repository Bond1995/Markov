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
run1 = api.run("linx/markov-LLM-2-full-layers/7k2yewfn") # l=1
df1 = run1.history()
loss1 = df1["val/loss"].values[:]
run2 = api.run("linx/markov-LLM-2-full-layers/cjspvplm") # l=4
df2 = run2.history()
loss2 = df2["val/loss"].values[:]
run3 = api.run("linx/markov-LLM-2-full-layers/4uvizmob") # l=8
df3 = run3.history()
loss3 = df3["val/loss"].values[:]
#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(loss1, color="tab:green", label="                                                    ", linewidth=2)
ax.plot(loss2, color="tab:orange", label="                                                    ", linewidth=2)
ax.plot(loss2, color="tab:purple", label="                                                    ", linewidth=2)
ax.axhline(y = 0.545, color="black", linestyle = '--', label="            ", linewidth=2, zorder=1)
ax.axhline(y = 0.673, color="black", linestyle = ':', label="            ", linewidth=2, zorder=1)
ax.set(xlabel=" ", ylabel=" ")
plt.locator_params(axis='y', nbins=6)
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
plt.xlim((0,100))
plt.ylim((0.5,0.8))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(prop={'size': 12}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("loss2-layers.pdf")