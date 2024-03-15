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
run1 = api.run("linx/markov-LLM-1-final/l57iuoud") # p + q > 1, without tying
df1 = run1.history()
loss1 = df1["val/loss"].values[:]
run2 = api.run("linx/markov-LLM-1-final/vufr9uoc") # p + q > 1, with tying
df2 = run2.history()
loss2 = df2["val/loss"].values[:]
#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
#ax.plot(loss1, color="tab:green", label="                                                    ", linewidth=3)
ax.plot(loss2, color="tab:red", label="                                                    ", linewidth=3)
ax.axhline(y = 0.619, color="black", linestyle = '--', label="            ", linewidth=3, zorder=1)
ax.axhline(y = 0.666, color="black", linestyle = ':', label="            ", linewidth=3, zorder=1)
ax.set(xlabel=" ", ylabel=" ")
plt.locator_params(axis='y', nbins=6)
ax.xaxis.label.set_fontsize(18)
ax.yaxis.label.set_fontsize(18)
plt.xlim((0,100))
plt.ylim((0.6,0.8))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax.legend(prop={'size': 18}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("loss2-only-tying.pdf")