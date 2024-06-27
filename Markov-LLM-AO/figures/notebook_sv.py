#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

api = wandb.Api()

# 
res1 = []
run = api.run("linx/markov-PQ/5kz5sqs9")
df = run.history()
c = df["batch10"].values[:]
print(c[~(np.isnan(c))])
#%%
# sns.set_style("whitegrid")

# fig, ax = plt.subplots()
# ax.plot(df["_step"], df["layer1.0.conv1.weight"], color="tab:blue", label="First hidden layer", linewidth=1.5)
# ax.plot(df["_step"], df["layer3.0.conv2.weight"], color="tab:orange", label="Central layer", linewidth=1.5)
# ax.plot(df["_step"], df["layer4.1.conv2.weight"], color="tab:green", label="Last hidden layer", linewidth=1.5)
# ax.set(xlabel="Epoch", ylabel="Fraction of energy in Top-8 components")
# ax.xaxis.label.set_fontsize(12)
# ax.yaxis.label.set_fontsize(12)
# ax.legend(prop={'size': 12})
# ax.grid(True, which="both")
# plt.xlim([1, 150])
# fig.savefig("cifar10-rank-study.pdf")