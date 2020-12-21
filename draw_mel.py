import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

files = ['transfer_synthesis_0.wav.pt',
         'transfer_synthesis_2.wav.pt',
         'transfer_synthesis_4.wav.pt']
datas = []

for file in files:
    data = torch.load(file)
    data = data.numpy()
    datas.append(data)

plt.subplots_adjust(wspace=0.5, hspace=0)
gs = gridspec.GridSpec(2,4)
ax = plt.subplot(gs[0, :2])
ax.imshow(datas[0])
ax = plt.subplot(gs[0, 2:])
ax.imshow(datas[1])
ax = plt.subplot(gs[1, 1:3])
ax.imshow(datas[2])

plt.show()
