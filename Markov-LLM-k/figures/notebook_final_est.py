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
est_p1 = df1["est/est_00"].values[:]
est_p1 = est_p1[~np.isnan(est_p1)]
run2 = api.run("linx/markov-LLM-1-final/tf87x432") # p + q < 1, with tying
df2 = run2.history()
est_p2 = df2["est/est_00"].values[:]
est_p2 = est_p2[~np.isnan(est_p2)]
#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.axhline(y = 0.2, color="black", linestyle = '--', label="                                      ", linewidth=3)
ax.plot(est_p1, color="tab:green", label="                              ", linewidth=3)
ax.plot(est_p2, color="tab:purple", label="                              ", linewidth=3)
ax.set(xlabel=" ", ylabel=" ")
plt.locator_params(axis='y', nbins=6)
ax.xaxis.label.set_fontsize(18)
ax.yaxis.label.set_fontsize(18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim((0,100))
plt.ylim((0.1,0.3))
ax.legend(prop={'size': 18})
ax.grid(True, which="both")
fig.savefig("est1.pdf")