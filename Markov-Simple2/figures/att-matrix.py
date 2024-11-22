import torch
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
att = np.load('att-mat-1000.pt.npy')
plt.imshow(att[:30, :30], cmap='Blues', interpolation='nearest')
plt.colorbar(location='top', shrink=0.6)
plt.savefig('att-mat.pdf', bbox_inches='tight')
