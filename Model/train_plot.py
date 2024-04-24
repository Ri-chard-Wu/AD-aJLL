"""  JLL  2022.3.1, 2023.8.28
(sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python train_plot.py
"""
import numpy as np
import matplotlib.pyplot as plt

plt.subplot(311)
train_loss = np.loadtxt('./saved_model/train_loss_out.txt')
tl, = plt.plot(train_loss, 'k--', label='train')
valid_loss = np.loadtxt('./saved_model/valid_loss_out.txt')
vl, = plt.plot(valid_loss, 'g--', label='valid')
plt.ylabel("loss")
legend1 = plt.legend([tl], ['train'], edgecolor='None', loc='upper right')
plt.legend([vl], ['valid'], edgecolor='None', loc='lower left')
plt.gca().add_artist(legend1)

plt.subplot(312)
train_rmse = np.loadtxt('./saved_model/train_rmse_out.txt')
tl, = plt.plot(train_rmse, 'k--', label='train')
valid_rmse = np.loadtxt('./saved_model/valid_rmse_out.txt')
vl, = plt.plot(valid_rmse, 'g--', label='valid')
plt.ylabel("rmse")
legend1 = plt.legend([tl], ['train'], edgecolor='None', loc='upper right')
plt.legend([vl], ['valid'], edgecolor='None', loc='lower left')
plt.gca().add_artist(legend1)

plt.subplot(313)
train_mae = np.loadtxt('./saved_model/train_mae_out.txt')
tl, = plt.plot(train_mae, 'k--', label='train')
valid_mae = np.loadtxt('./saved_model/valid_mae_out.txt')
vl, = plt.plot(valid_mae, 'g--', label='valid')
plt.ylabel("mae")
plt.xlabel("epoch")
legend1 = plt.legend([tl], ['train'], edgecolor='None', loc='upper right')
plt.legend([vl], ['valid'], edgecolor='None', loc='lower left')
plt.gca().add_artist(legend1)

plt.draw()
plt.savefig('./saved_model/modelB6_outs.png')
plt.pause(0.5)
input("Press ENTER to close ...")
plt.close()
