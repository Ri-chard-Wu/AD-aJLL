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


from tqdm import tqdm

import glob
import pickle

# from serverB6 import client_generator, BATCH_SIZE, STEPS, EPOCHS

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

 
# print(f"## (len(tf.config.experimental.list_physical_devices('GPU'))): {(len(tf.config.experimental.list_physical_devices('GPU')))}")



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
    'lr_min': 1e-4,

    'accum_steps': 1,
    'batch_size': 2,

    'weight_decay': 0.004,
    'ema_momentum': 0.99, #0.99,

    'host': "localhost",
    'port': 5557,
    'port_val': 5558,

    'log_interval': 20,
    'save_interval': 100,
    'validate_interval': 1000,

    # 'ckpt_load_path': f'ckpt/modelB6-{700}.h5',
  }
) 



def normalize_img(imgs):

 
    assert imgs.shape[1] == 12

    c = 12

    mean = [np.mean(imgs[:, i, ...]) for i in range(c)]
    std = [np.std(imgs[:, i, ...]) for i in range(c)]
  

    for i in range(c):
        imgs[:, i, ...] = (imgs[:, i, ...] - mean[i]) / std[i]

    return imgs



def get_data(pkl_files):

    file_idx = 0
    while(1):

        pkl_file = pkl_files[file_idx]
        # print(f'## get data: {pkl_file}')
        file_idx += 1
        if(file_idx >= len(pkl_files)): file_idx = 0
 
            
        with open(pkl_file, 'rb') as f:      

            data = pickle.load(f)   

            n = data['Ximgs'].shape[0]
            
            assert data['Ximgs'].shape == (n, 12, 128, 256)
            assert data['Xin1'].shape == (n, 8)
            assert data['Xin2'].shape == (n, 2)
            assert data['Xin3'].shape == (n, 512)
            

            Y_shapes = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]
            Y_dim = sum(Y_shapes)

            for i in range(12):
                assert data['Y'][i].shape == (n, Y_shapes[i])

 
            
            # print(f"bn mean: {np.mean(data['Ximgs'], axis=(0,2,3))}, std: {np.std(data['Ximgs'], axis=(0,2,3))}")

            data['Ximgs'] = normalize_img(data['Ximgs'])

            # print(f"an mean: {np.mean(data['Ximgs'], axis=(0,2,3))}, std: {np.std(data['Ximgs'], axis=(0,2,3))}")
            # exit()

        
        n_batches = n // para.batch_size
        for b in range(n_batches):

            i1 = b * para.batch_size
            i2 = (b+1) * para.batch_size
            if(i2 > n): break
             

            Xins  = [data['Ximgs'][i1:i2], data['Xin1'][i1:i2], data['Xin2'][i1:i2], data['Xin3'][i1:i2]]
            Xins = [tf.convert_to_tensor(x, dtype=tf.float32) for x in Xins]

            # Ytrue = np.hstack(tup[4:])
            Ytrue = [data['Y'][i][i1:i2] for i in range(12)]
            Ytrue = np.hstack(Ytrue)
            assert Ytrue.shape == (para.batch_size, Y_dim)
            Ytrue = tf.convert_to_tensor(Ytrue, dtype=tf.float32)
        
            # print(f'## yield batch: {b} of file {file_idx}')
            yield Xins, Ytrue






def maxae(y_true, y_pred):
  return KB.max(KB.abs(y_pred - y_true), axis=-1)



PATH_IDX   = 0      
LL_IDX     = 385    
RL_IDX     = 771   
LEAD_IDX   = 1157 

def train_loss_fn(y_true, y_pred):
  
  # y_true = y_true[:, PATH_IDX:   LEAD_IDX]
  # y_pred = y_pred[:, PATH_IDX:   LEAD_IDX]


  # loss_CS = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)  # MC4 (Md Case 4)
  loss_MSE = tf.keras.losses.mse(y_true, y_pred)  # MC5
  
  # loss = 0.5 * loss_CS + 0.5 * loss_MSE  # MC5
 
  return loss_MSE






model = get_model()


if 'ckpt_load_path' in para:
    model.load_weights(para.ckpt_load_path)  # for retraining
  
optimizer = tf.keras.optimizers.AdamW(learning_rate=para.base_lr,
                  weight_decay=para.weight_decay, ema_momentum=para.ema_momentum, clipvalue=1.0)        

# train_dataset = get_data(20, para.host, port=para.port)
# val_dataset = get_data(20, para.host, port=para.port_val)


pkl_files = glob.glob("/home/richard/dataB6/*/data.pkl")
assert len(pkl_files) == 3
print(f'pkl_files: {pkl_files}')


train_dataset = get_data(pkl_files[0:3])
# val_dataset = get_data(pkl_files[2:3])


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


 
@tf.function
def _train_step(X_step, Y_step):

    with tf.GradientTape() as tape:
        Y_pred = model(X_step, training=True) # (128, 10)
        loss = train_loss_fn(Y_step, Y_pred)
        loss = tf.reduce_mean(loss)
         
    grad = tape.gradient(loss, model.trainable_variables)

    metric = tf.reduce_mean(maxae(Y_step, Y_pred))

    return loss, metric, grad





with open("log.txt", "w") as f: f.write("")

 
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
          
            # print(f"## (len(tf.config.experimental.list_physical_devices('GPU'))): {(len(tf.config.experimental.list_physical_devices('GPU')))}")
            X_step = [a[step*step_size:(step+1)*step_size] for a in X]
            Y_step = Y[step*step_size:(step+1)*step_size]
 
            loss, metric, grad = _train_step(X_step, Y_step)
 
            for i in range(len(accum_gradients)):
                 
                accum_gradients[i] += grad[i]
            
             
            total_loss += loss.numpy() 
            total_metric += metric.numpy() 

       
        averaged_gradients = [accum_grad / tf.cast(para.accum_steps, tf.float32) for accum_grad in accum_gradients]
        # clipped_grads, _ = tf.clip_by_global_norm(averaged_gradients, para.grad_norm_clip)
        optimizer.apply_gradients(zip(averaged_gradients, model.trainable_variables))

     
        update_counts += 1
        lr_next = lr_scheduler(update_counts)
        tf.keras.backend.set_value(optimizer.learning_rate, lr_next)


        if update_counts % para.log_interval == 0:
            log = f'[{update_counts}] train loss: {total_loss / para.accum_steps}' + \
            f', train metric: {total_metric / para.accum_steps}' + f', lr: {lr_next}' 
            print(log)
            with open("log.txt", "a") as f: f.write(log + '\n')

        

        # if update_counts % para.validate_interval == 0:
        #     loss, metric = validate()                
        

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

