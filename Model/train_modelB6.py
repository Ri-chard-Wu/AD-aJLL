 




import os
import cv2
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

from data import RGB_to_YUV, Y_shape, plot_outs

 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

 
# print(f"## (len(tf.config.experimental.list_physical_devices('GPU'))): {(len(tf.config.experimental.list_physical_devices('GPU')))}")
 

 
# cap = cv2.VideoCapture('/home/richard/Downloads/TData1/THD--2020-04-28--19-22-07--10--fcamera.hevc')

# ret, frame = cap.read()  # frame: (874, 1164, 3) bgr img.

# print(f'type(frame): {type(frame)}')
# print(f'frame.dtype: {frame.shape}')
# exit()



class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]

 
para = AttrDict(
  {

    'total_steps': 1000,
    'base_lr': 1e-3,
    'decay_type': 'cosine',
    'warmup_steps': 5,
    'grad_norm_clip': 1,
    'lr_min': 1e-4,

    'accum_steps': 16,
    'batch_size': 32,
    'file_batch_size': 2,

    'weight_decay': 0.004, # 0.05
    'ema_momentum': 0.99, #0.99,

    'log_interval': 1,
    'save_interval': 25,
    'validate_interval': 1,

    # 'ckpt_load_path': f'ckpt/modelB6-{2100}.h5',
  }
) 



 
PATH_DISTANCE = 192
LANE_OFFSET = 1.8
LEAD_X_SCALE = 10   # x_scale in driving.cc
LEAD_Y_SCALE = 10   # y_scale in driving.cc
LEAD_V_SCALE = 1


# def normalize_img(imgs):

 
#     assert imgs.shape[1] == 12

#     c = 12

#     mean = [np.mean(imgs[:, i, ...]) for i in range(c)]
#     std = [np.std(imgs[:, i, ...]) for i in range(c)]
  

#     for i in range(c):
#         imgs[:, i, ...] = (imgs[:, i, ...] - mean[i]) / std[i]

#     return imgs


def get_frames(hevc_file):

    frames = []

    cap = cv2.VideoCapture(hevc_file)

    ret, frame = cap.read()  # frame: (874, 1164, 3) bgr img in uint8 np array.
     
    while ret:        
        frames.append(frame)
        ret, frame = cap.read()  
    return frames


 

def sample_data(file_batch, num_sample, debug=False):
 

    N = num_sample * len(file_batch)

    X  = [np.zeros((N, 12, 128, 256)), np.zeros((N, 8)), np.zeros((N, 2)), np.zeros((N, 512))]
    if (debug): RGBs = np.zeros((N, 874, 1164, 3)) # for debug.
    else: RGBs = None
    Y = [np.zeros((N, s)) for s in Y_shape]


    for fidx, (pkl_file, hevc_file) in enumerate(file_batch):
        
        print(f'fidx: {fidx}')
        
        frames = get_frames(hevc_file)
        
        idxes = np.random.choice(np.arange(len(frames)-1), size=num_sample, replace=False)   
        # frames = [frames[i] for i in idxes]
        
        with open(pkl_file, 'rb') as f:      

            data = pickle.load(f)   
            assert data['Xin3'].shape[1] == 512
       
            for i in range(12):
                assert data['Y'][i].shape[1] == Y_shape[i]

 
        for i in range(num_sample):
             
            idx = idxes[i]  

            X[3][i + fidx*num_sample] = data['Xin3'][idx]
            X[0][i + fidx*num_sample] = np.vstack((RGB_to_YUV(frames[idx]), RGB_to_YUV(frames[idx+1]))) # (12, 128, 256).

            for j in range(12):
                Y[j][i + fidx*num_sample] = data['Y'][j][idx]
         

            if (debug): RGBs[i + fidx*num_sample] = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)


    idxes = np.random.choice(np.arange(N), size=N, replace=False)   

    X[0] = tf.convert_to_tensor(X[0][idxes], dtype=tf.float32)
    X[3] = tf.convert_to_tensor(X[3][idxes], dtype=tf.float32)
    if (debug): RGBs = RGBs[idxes]
    for i in range(12):
        Y[i] = tf.convert_to_tensor(Y[i][idxes], dtype=tf.float32)

    X = [tf.convert_to_tensor(x, dtype=tf.float32) for x in X]

    return  X, Y, RGBs

        
    


def get_data(pkl_files, file_batch_size, batch_size, debug=False, num_sample = 30):

    print(f'## n data: {len(pkl_files)}')   

    hevc_files = [pkl_file.replace('data.pkl', 'fcamera.hevc') for pkl_file in pkl_files] 
   
    FB = file_batch_size
    n_FB = len(pkl_files) // FB 
    
    B = batch_size
    n_B = (num_sample * FB) // B 
        
    N = num_sample * FB

    while(1): 

        for fb in range(n_FB):
 
            if((fb+1) * FB > len(pkl_files)): break
 
            file_batch = [(pkl_files[i], hevc_files[i]) for i in range(fb * FB, (fb+1) * FB)]

            
            X, Y, RGBs = sample_data(file_batch, num_sample, debug)
            


            for b in range(n_B):
 
                if((b+1) * B > N): break
  
                # Ximg = np.concatenate([X[0][i][None] for i in range(b*B, (b+1)*B)])
                # assert Ximg.shape == (B, 12, 128, 256)

                X_batch = []
                for i in range(4):
                    X_batch.append(X[i][b*B:(b+1)*B])
                
                Y_batch = []
                for i in range(12):
                    Y_batch.append(Y[i][b*B:(b+1)*B])
                # Y_batch = np.hstack(Y_batch)


                RGB_batch = RGBs[b*B:(b+1)*B] if debug else None
                # print(f'## yield batch: {b} of file {file_idx}')
                yield X_batch, Y_batch, RGB_batch


 

def maxae(y_true, y_pred):
  # [for i in range(len(y_true))]
  return KB.max(KB.abs(y_pred - y_true), axis=-1)

 

def parse_outs(y):

    # Y_shape = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]

    # p, ll, rl, lead, long_x, long_v, long_a, desire_state, meta, desire_pred, pose, state = y
    
    p, ll, rl, lead = y[:4]
   

    path = p[:, :PATH_DISTANCE] + 0.1
    path_stds = tf.math.softplus(p[:, PATH_DISTANCE:PATH_DISTANCE*2])    
    path_valid_len = tf.clip_by_value(p[:, PATH_DISTANCE*2], 5, PATH_DISTANCE)     

    lll = ll[:, :PATH_DISTANCE] + LANE_OFFSET + 0.1
    lll_stds = tf.math.softplus(ll[:, PATH_DISTANCE:PATH_DISTANCE*2])
    lll_valid_len = tf.clip_by_value(ll[:, PATH_DISTANCE*2], 5, PATH_DISTANCE) 
    lll_prob = tf.math.sigmoid(ll[:, PATH_DISTANCE*2 + 1])

    rll = rl[:, :PATH_DISTANCE] - LANE_OFFSET - 0.1
    rll_stds = tf.math.softplus(rl[:, PATH_DISTANCE:-2])
    rll_valid_len = tf.clip_by_value(rl[:, -2], 5, PATH_DISTANCE) 
    rll_prob = tf.math.sigmoid(rl[:, -1])

    lead_prob = tf.math.sigmoid(lead[:, -3]) # lead.shape = (b, 58)
    lead = tf.reshape(lead[:, :-3], (-1, 5, 11)) # (b, 5, 11)
    lead_weights = tf.nn.softmax(lead[:, :, 8]) # lead_weights.shape = (b, 5)
    lidx = tf.math.argmax(lead_weights, axis=-1, output_type=tf.dtypes.int32)   # (b,)
    
    # print(f'lidx.shape: {lidx.shape}, para.batch_size: {para.batch_size}')

    idxes = tf.stack([tf.range(tf.shape(lidx)[0]), lidx], axis=1)
    
    lead_max = tf.gather_nd(lead, idxes) # (b, 11)

    # scales = [LEAD_X_SCALE, LEAD_Y_SCALE, LEAD_V_SCALE, 1]
    # lead_xyva = np.column_stack([lead_max[:, i] * scales[i] for i in range(4)])

    lead_xyva = tf.stack([lead_max[:, 0] * LEAD_X_SCALE,
                                lead_max[:, 1] * LEAD_Y_SCALE,
                                lead_max[:, 2] * LEAD_V_SCALE,
                                lead_max[:, 3]], axis=-1)                          
    
    lead_xyva_std = tf.stack([tf.math.softplus(lead_max[:, 4]) * LEAD_X_SCALE,
                                tf.math.softplus(lead_max[:, 5]) * LEAD_Y_SCALE,
                                tf.math.softplus(lead_max[:, 6]) * LEAD_V_SCALE,
                                tf.math.softplus(lead_max[:, 7])], axis=-1)


    # lead_xyva_std = np.column_stack([tf.math.softplus(lead_max[:, 4+i]) * scales[i] for i in range(4)])

    return path, path_stds, path_valid_len, lll, lll_stds, lll_valid_len, lll_prob, \
        rll, rll_stds, rll_valid_len, rll_prob, lead_xyva, lead_xyva_std, *y[4:]

 

def train_loss_fn(y_true, y_pred):
    

    # # print(f'len(y_true): {len(y_true)}, type(y_true): {type(y_true)}')
    # # print(f'len(y_pred): {len(y_pred)}, type(y_pred): {type(y_pred)}')

    # y_true = parse_outs(y_true)
    # y_pred = parse_outs(y_pred)

    # # for i in tf.range(x):  # Use tf.range() for TensorFlow's tensors
    # #     sum_value += tf.cast(i, tf.float32)

    # means = [tf.math.reduce_mean(y) for y in y_true]
    # stds = [tf.math.reduce_std(y) for y in y_true]


    # y_true = [(y_true[i] - means[i]) / stds[i] for i in range(len(y_true))]
    # y_pred = [(y_pred[i] - means[i]) / stds[i] for i in range(len(y_true))]
  
    # total_loss = 0
    # for i in range(len(y_true)): 
    #     total_loss += tf.keras.losses.mse(y_true[i], y_pred[i]) # (b,)
    #     # total_loss += tf.clip_by_value(loss, 0, 1) 

    # print(f'y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}')
    total_loss = tf.keras.losses.mse(y_true, y_pred)
  
    return tf.math.reduce_mean(total_loss)





def validate(n=20):
    
    # print(f'## validating...') 
  
    losses = 0
    metrics = 0
    # count = 0
    for i, batch in tqdm(enumerate(val_dataset)):
        if i >= n: break

        X, Y, RGBs = batch
        # count += len(Y)

        Y_pred = model(X, training=False) # a list.

        Y_pred_cat = np.hstack(Y_pred)
        Y_cat = np.hstack(Y)

        loss = train_loss_fn(Y_cat, Y_pred_cat)
        metric = tf.reduce_mean(maxae(Y_cat, Y_pred_cat))

        for j in range(len(RGBs)):
            frame = RGBs[j] # (874, 1164, 3)            
            # plot_outs([y.numpy()[j:j+1] for y in Y_pred], frame, dir_name=f'output/val', file_name=f'pred-{i}-{j}.png')
            plot_outs([y.numpy()[j:j+1] for y in Y], frame, dir_name=f'output/val', file_name=f'true-{i}-{j}.png')
 
        losses += loss.numpy().sum()
        metrics += metric.numpy().sum()

    # assert count == n * para.batch_size, f'{count} != {n * para.batch_size}'

    losses = losses / n
    metrics = metrics / n

    # print(f'val loss: {losses}, val metric: {metrics}, count: {count}')

    return losses, metrics





def lr_scheduler(step):

    step = step % para.total_steps

    lr = para.base_lr

    progress = (step - para.warmup_steps) / float(para.total_steps - para.warmup_steps)
    progress = np.clip(progress, 0.0, 1.0)
  
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    
    if para.warmup_steps:
        lr = lr * np.minimum(1., step / para.warmup_steps)

    lr = max(lr, para.lr_min)

    return np.asarray(lr, dtype=np.float32) 




 
@tf.function
def _train_step(X_step, Y_step):


    Y_step = tf.concat(Y_step, axis=-1)
    

    with tf.GradientTape() as tape:
        
        Y_pred = model(X_step, training=True) # Y_pred: list. 

        Y_pred = tf.concat(Y_pred, axis=-1)

        loss = train_loss_fn(Y_step, Y_pred)        
        # loss = tf.reduce_mean(loss)
         
    grad = tape.gradient(loss, model.trainable_variables)

    
    metric = tf.reduce_mean(maxae(Y_step, Y_pred))
   
    return loss, metric, grad





 




model = get_model()


if 'ckpt_load_path' in para:
    model.load_weights(para.ckpt_load_path)  # for retraining
  
optimizer = tf.keras.optimizers.AdamW(learning_rate=para.base_lr,
                  weight_decay=para.weight_decay, ema_momentum=para.ema_momentum, clipvalue=1.0)        

 
all_pkl = glob.glob("/home/richard/Downloads/TData1/*.pkl") 
split = int(len(all_pkl) * 0.8)
train_pkl = all_pkl[:split]
val_pkl = all_pkl[split:]

train_dataset = get_data(train_pkl, para.file_batch_size, para.batch_size, debug=False)
val_dataset = get_data(val_pkl, 1, 2, debug=True)


# exit()


with open("log.txt", "w") as f: f.write("")
with open("val.txt", "w") as f: f.write("")

 
update_counts = 0
for epoch in range(10000):


    for i, batch in enumerate(train_dataset): # have ~600 training batches.

        # X, Y, _ = batch                    

        # total_loss = 0.0
        # total_metric = 0.0

        # accum_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
        # step_size = para.batch_size // para.accum_steps
        
        # for step in range(para.accum_steps): 

        #     print(f'step: {step}')
          
        #     # print(f"## (len(tf.config.experimental.list_physical_devices('GPU'))): {(len(tf.config.experimental.list_physical_devices('GPU')))}")
        #     X_step = [a[step*step_size:(step+1)*step_size] for a in X]
        #     Y_step = [a[step*step_size:(step+1)*step_size] for a in Y]
        #     # Y_step = Y[step*step_size:(step+1)*step_size]
            
         
 
        #     loss, metric, grad = _train_step(X_step, Y_step)

            
 
        #     for i in range(len(accum_gradients)):
                 
        #         accum_gradients[i] += grad[i]
            

        #     # print(f'## a, loss: {loss}')
        #     total_loss += loss.numpy() 

        #     # print(f'## b, metric: {metric}')

        #     total_metric += metric.numpy() 

        #     # print(f'## c')

 
        # averaged_gradients = [accum_grad / tf.cast(para.accum_steps, tf.float32) for accum_grad in accum_gradients]
        # # clipped_grads, _ = tf.clip_by_global_norm(averaged_gradients, para.grad_norm_clip)
        # optimizer.apply_gradients(zip(averaged_gradients, model.trainable_variables))

     
        # update_counts += 1
        # lr_next = lr_scheduler(update_counts)
        # tf.keras.backend.set_value(optimizer.learning_rate, lr_next)



        # if update_counts % para.log_interval == 0:
        #     log = f'[{update_counts}] train loss: {total_loss / para.accum_steps}' + \
        #                   f', train metric: {total_metric / para.accum_steps}' + f', lr: {lr_next}' 
        #     print(log)
        #     with open("log.txt", "a") as f: f.write(log + '\n')

        

        if update_counts % para.validate_interval == 0:
            print(f'validating...') 
            loss, metric = validate()
            log = f'val loss: {loss}, val metric: {metric}'
            print(log)
            with open("val.txt", "a") as f: f.write(log + '\n')                
        

        if update_counts % para.save_interval == 0:
            model.save_weights(f'ckpt/modelB6-{update_counts}.h5')
            print(f'saved ckpt: ckpt/modelB6-{update_counts}.h5')

  


 