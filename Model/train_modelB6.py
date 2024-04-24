"""   YPL, YJW, JLL, 2021.9.8 - 2024.2.21
for 230826, 240124, 240207
from /home/jinn/openpilot/aJLLold/Model/train_modelB6b.py

1. Use the output of supercombo079.keras as ground truth data to train modelB6
2. Tasks: Path Prediction + Lane Detection + Lead Car Detection
   y_true[2383] = (Ytrue0, Ytrue1, Ytrue2, ..., Ytrue10, Ytrue11)
   y_pred[2383] = outs[0] + ... + outs[11] in sim_output.txt.

Input:
  /home/jinn/dataB6/UHD--2018-08-02--08-34-47--32/yuv.h5
  /home/jinn/dataB6/UHD--2018-08-02--08-34-47--33/yuv.h5
  /home/jinn/dataB6/UHD--2018-08-02--08-34-47--37/yuv.h5
Output:
  /home/jinn/openpilot/aJLL/Model/saved_model/B6.keras
  /home/jinn/openpilot/aJLL/Model/saved_model/B6BW.hdf5
  /home/jinn/openpilot/aJLL/Model/output/B6Loss.png

Train and Validate Model: Run on 3 terminals
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python serverB6.py --port 5557
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python serverB6.py --port 5558 --validation
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python train_modelB6.py --port 5557 --port_val 5558

Test Model Step 1: /output/B6Sim.png
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python simulatorB6.py
Test Model Step 2: /output/B6Y.png
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python train_modelB6.py

Important Hyper-Parameters (tune these for better results):
BATCH_SIZE, STEPS, EPOCHS, learning_rate, decay_steps, decay_rate, weight_decay, clipvalue

Set BATCH_SIZE, STEPS, EPOCHS in serverB6.py. Set larger BATCH_SIZE if you have more GPUs.
Epochs (JLL) != EPOCHS (Keras); Steps (JLL) != STEPS (Keras); Run datagen_debug() for these.
"""
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as KB
 
from modelB6 import get_model
from serverB6 import client_generator, BATCH_SIZE, STEPS, EPOCHS

from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]

# para.
para = AttrDict(
  {

    'total_steps': 1000,
    'base_lr': 1e-3,
    'decay_type': 'cosine',
    'warmup_steps': 5,
    'grad_norm_clip': 1,
    'lr_min': 1e-5,

    'accum_steps': 1,
    'batch_size': BATCH_SIZE,

    'weight_decay': 0.004,
    'ema_momentum': 0.99,

    'host': "localhost",
    'port': 5557,
    'port_val': 5558,

    'log_interval': 5,
    'save_interval': 30,
    'validate_interval': 60

    # 'ckpt_load_path': f'ckpt/modelB6-{1050}.h5',
  }
) 


def get_data(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    Ximgs_rgb, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4,\
                           Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11 = tup
 
    Xins  = [*tup[:4]]
    Xins = [tf.convert_to_tensor(x, dtype=tf.float32) for x in Xins]

    Ytrue = np.hstack(tup[4:])
    Ytrue = tf.convert_to_tensor(Ytrue, dtype=tf.float32)
 
    yield Xins, Ytrue



def maxae(y_true, y_pred):
  return KB.max(KB.abs(y_pred - y_true), axis=-1)




def custom_loss(y_true, y_pred):
  
  loss_CS = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)  # MC4 (Md Case 4)
  loss_MSE = tf.keras.losses.mse(y_true, y_pred)  # MC5
  loss = 0.5 * loss_CS + 0.5 * loss_MSE  # MC5

  metric = maxae(y_true, y_pred)

  return loss, metric






model = get_model()


if 'ckpt_load_path' in para:
    model.load_weights(para.ckpt_load_path)  # for retraining
  
optimizer = tf.keras.optimizers.AdamW(learning_rate=para.base_lr,
                  weight_decay=para.weight_decay, ema_momentum=para.ema_momentum)        

train_dataset = get_data(20, para.host, port=para.port)
val_dataset = get_data(20, para.host, port=para.port_val)


def validate(n=5):
    
    print(f'## validating...') 
  
    losses = 0
    metrics = 0
    count = 0
    for i, batch in tqdm(enumerate(val_dataset)):
        if i >= n: break

        X, Y = batch
        count += len(Y)

        Y_pred = model(X, training=False) # (128, 10)
        loss, metric = train_loss_fn(Y, Y_pred)

        losses += loss.numpy().sum()
        metrics += metric.numpy().sum()

    assert count == n * para.batch_size, f'{count} != {n * para.batch_size}'

    losses = losses / count
    metrics = metrics / count

    print(f'val loss: {losses}, val metric: {metrics}, count: {count}')

    return losses, metrics





def lr_scheduler(step):

    lr = para.base_lr

    progress = (step - para.warmup_steps) / float(para.total_steps - para.warmup_steps)
    progress = np.clip(progress, 0.0, 1.0)

    if para.decay_type == 'linear':
        lr = linear_end + (lr - linear_end) * (1.0 - progress)
    elif para.decay_type == 'cosine':
        lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    else:
        raise ValueError(f'Unknown lr type {decay_type}')

    if para.warmup_steps:
        lr = lr * np.minimum(1., step / para.warmup_steps)

    lr = max(lr, para.lr_min)

    return np.asarray(lr, dtype=np.float32) 


train_loss_fn = custom_loss


@tf.function
def _train_step(X_step, Y_step):

    with tf.GradientTape() as tape:
        Y_pred = model(X_step, training=True) # (128, 10)
        loss, metric = train_loss_fn(Y_step, Y_pred)
        loss = tf.reduce_mean(loss)
        metric = tf.reduce_mean(metric)

    grad = tape.gradient(loss, model.trainable_variables)

    return loss, metric, grad





update_counts = 0
for epoch in range(10000):


    for i, batch in enumerate(train_dataset): # have ~600 training batches.

        X, Y = batch
                    
        if(len(X[0]) < para.batch_size): break


        total_loss = 0.0
        total_metric = 0.0

        accum_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
        step_size = para.batch_size // para.accum_steps
        
        for step in range(para.accum_steps): 
            
            X_step = [a[step*step_size:(step+1)*step_size] for a in X]
            Y_step = Y[step*step_size:(step+1)*step_size]
 
            loss, metric, grad = _train_step(X_step, Y_step)
           
            for i in range(len(accum_gradients)):
                 
                accum_gradients[i] += grad[i]
            

            total_loss += loss.numpy() 
            total_metric += metric.numpy() 

        averaged_gradients = [accum_grad / tf.cast(para.accum_steps, tf.float32) for accum_grad in accum_gradients]
        clipped_grads, _ = tf.clip_by_global_norm(averaged_gradients, para.grad_norm_clip)
        optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))


        update_counts += 1
        lr_next = lr_scheduler(update_counts)
        tf.keras.backend.set_value(optimizer.learning_rate, lr_next)


        if update_counts % para.log_interval == 0:
            print(f'[{update_counts}] train loss: {total_loss / para.accum_steps}' + \
            f', train metric: {total_metric / para.accum_steps}' + f', lr: {lr_next}')


        if update_counts % para.validate_interval == 0:
            loss, metric = validate()                
        

        if update_counts % para.save_interval == 0:
            model.save_weights(f'ckpt/modelB6-{update_counts}.h5')

  




# print('########### a')
# model.save('./saved_model/B6.keras')  # rename and save "long" trained .keras

# print('########### b')
# end = time.time()
# hours, rem = divmod(end-start, 3600)
# minutes, seconds = divmod(rem, 60)
# print("#--- Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

# print('########### c')
# # Plot training results
# plt.subplot(311)

# print('########### d')

# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.ylabel("loss")
# plt.legend(['train', 'validate'], loc='upper right')

# print('########### e')

# plt.subplot(312)
# plt.plot(history.history["maxae"])
# plt.plot(history.history["val_maxae"])
# plt.ylabel("maxae")
# plt.xlabel("epoch")
# plt.legend(['train', 'validate'], loc='upper right')

# plt.subplot(313)
# plt.plot(history.history["lr"])  
# plt.ylabel("learning rate")

# plt.draw()
# plt.savefig('./output/B6Loss.png')
# plt.pause(0.5)
# input("Press ENTER to close ...")
# plt.close()

