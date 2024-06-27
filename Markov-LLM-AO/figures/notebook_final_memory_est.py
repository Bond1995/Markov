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
run1 = api.run("linx/markov-LLM-2-final-memory/w1ntj45p") # memory 2
df1 = run1.history()
est_p1 = df1["est/est_00"].values[:]
est_p1 = est_p1[~np.isnan(est_p1)]
run2 = api.run("linx/markov-LLM-2-final-memory/3fcfnizf") # memory 8
df2 = run2.history()
est_p2 = df2["est/est_00"].values[:]
est_p2 = est_p2[~np.isnan(est_p2)]
run3 = api.run("linx/markov-LLM-2-final-memory/qts3aw7l") # memory 32
df3 = run3.history()
est_p3 = df3["est/est_00"].values[:]
est_p3 = est_p3[~np.isnan(est_p3)]
run4 = api.run("linx/markov-LLM-2-final-memory/xhatwjil") # memory 40
df4 = run4.history()
est_p4 = df4["est/est_00"].values[:]
est_p4 = est_p4[~np.isnan(est_p4)]
#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.axhline(y = 0.2, color="black", linestyle = '--', label="                 ", linewidth=1.5)
ax.plot(est_p1, color="tab:green", label="   ", linewidth=1.5)
ax.plot(est_p2, color="tab:blue", label="  ", linewidth=1.5)
ax.plot(est_p3, color="tab:orange", label="     ", linewidth=1.5)
ax.plot(est_p4, color="tab:purple", label="   ", linewidth=1.5)
ax.set(xlabel=" ", ylabel=" ")
#plt.locator_params(axis='y', nbins=6)
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim((0,100))
#plt.ylim((0.1,0.3))
ax.legend(prop={'size': 12})
ax.grid(True, which="both")
fig.savefig("memory-est.pdf")