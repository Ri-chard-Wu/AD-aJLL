 




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

    'accum_steps': 4,
    'batch_size': 8,
    'horizon': 512,
    'horizon_val': 512,
    

    'weight_decay': 0.004, # 0.05, 0.004
    'ema_momentum': 0.99, #0.99,

    'log_interval': 20,
    'save_interval': 2000,
    'validate_interval': 2000, 

    'ckpt_load_path': f'ckpt/best2.h5',
  }
) 

  

 



def read_frames(hevc_file):

    frames = []

    cap = cv2.VideoCapture(hevc_file)

    ret, frame = cap.read()  # frame: (874, 1164, 3) bgr img in uint8 np array.
     
    while ret:        
        frames.append(frame)
        ret, frame = cap.read()  

    cap.release()
    return frames
 


def gen_episodes_train(pkl_files):
  
    hevc_files = [pkl_file.replace('data.pkl', 'fcamera.hevc') for pkl_file in pkl_files]    

    H = para.horizon 
    B = para.batch_size
    n_B = len(pkl_files) // B 
         
    reps = 1

    while(1): 

        gfidx = 0

        for bidx in range(n_B):

            if((bidx+1) * B > len(pkl_files)): break
 
            for ridx in range(reps): 
    
                X0 = np.zeros((B, H, 12, 128, 256), dtype=np.uint8)            
                X3 = np.zeros((B, H, 512), dtype=np.float32)    
                Y = [np.zeros((B, H, s)) for s in Y_shape]
    
                progress_bar = tqdm(total=B, desc="sampleing training data...")
                
     
                
                for fidx in range(B):
                    
                    progress_bar.update(1)


                    while(1):
                        pkl_file = pkl_files[gfidx]
                        hevc_file = hevc_files[gfidx]                    
                        frames = read_frames(hevc_file) 

                        gfidx += 1
                        if(gfidx >= len(pkl_files)): gfidx = 0

                        if(len(frames)-H > 0):  
                            break
                        else:
                            print(f'[Warning] len(frames)-H <= 0): {hevc_file}')                        
                            continue


                    
                    
                    t0 = np.random.choice(np.arange(len(frames)-H), size=1)[0]
                    frames = frames[t0:t0+H+1]
    
                    with open(pkl_file, 'rb') as f:      

                        data = pickle.load(f)   
                        assert data['Xin3'].shape[1] == 512
                           
                        for i in range(12):
                            assert data['Y'][i].shape[1] == Y_shape[i]
    
                    for t in range(H):   
            
                        X0[fidx, t] = np.vstack((RGB_to_YUV(frames[t]), RGB_to_YUV(frames[t+1])))
                        X3[fidx, t] = data['Xin3'][t0+t]
                        for j in range(12):
                            Y[j][fidx, t] = data['Y'][j][t0+t]


                    # fidx += 1

                yield X0, X3, Y

                del X0, Y




def gen_episodes_val(pkl_files):
  
    hevc_files = [pkl_file.replace('data.pkl', 'fcamera.hevc') for pkl_file in pkl_files]    

    H = para.horizon_val
  
    while(1):

        idx = np.random.choice(np.arange(len(pkl_files)), size=1)[0]
        pkl_file = pkl_files[idx]
        hevc_file = hevc_files[idx]

        RGBs = np.zeros((1, H, 874, 1164, 3), dtype=np.uint8) # for debug.
        X0 = np.zeros((1, H, 12, 128, 256), dtype=np.uint8)   
        X3 = np.zeros((1, H, 512), dtype=np.float32)             
        Y = [np.zeros((1, H, s)) for s in Y_shape]
     
        frames = read_frames(hevc_file) 
        
        if(len(frames)-H <= 0): 
            print(f'[Warning] len(frames)-H <= 0): {hevc_file}')
            continue

        t0 = np.random.choice(np.arange(len(frames)-H), size=1)[0]

        

        frames = frames[t0:t0+H+1]
 
        with open(pkl_file, 'rb') as f:      

            data = pickle.load(f)    
            assert data['Xin3'].shape[1] == 512
            for i in range(12):
                assert data['Y'][i].shape[1] == Y_shape[i]

        for t in range(H):   

            RGBs[0, t] = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)

            X3[0, t] = data['Xin3'][t0+t]
            X0[0, t] = np.vstack((RGB_to_YUV(frames[t]), RGB_to_YUV(frames[t+1])))

            for j in range(12):
                Y[j][0, t] = data['Y'][j][t0+t]


        yield X0, X3, Y, RGBs

        del X0, X3, Y, RGBs

 


def validate(n=3):
        
    H = para.horizon_val
 
    losses_ar = []
    losses_spv = []
    metrics_ar = []
    metrics_spv = []
        
    for i, episode in tqdm(enumerate(val_episodes)):

        if i >= n: break

        X0, X3, Y, RGBs = episode
 

        X1_batch = tf.convert_to_tensor(np.zeros((1, 8)), dtype=tf.float32)
        X2_batch = tf.convert_to_tensor(np.zeros((1, 2)), dtype=tf.float32)
        rnn_st_batch = tf.convert_to_tensor(np.zeros((1, 512)), dtype=tf.float32)

        progress_bar = tqdm(total=H, desc=f"executing val episodes {i+1} / {n}...")
          

        try:
            for t in range(H):
    
                progress_bar.update(1)
        
                X0_batch = tf.convert_to_tensor(X0[:, t, ...], dtype=tf.float32)
                X3_batch = tf.convert_to_tensor(X3[:, t, ...], dtype=tf.float32)            
                Y_batch = [tf.convert_to_tensor(y[:, t, ...], dtype=tf.float32) for y in Y]
                
                
                Y_pred_spv_batch = model([X0_batch, X1_batch, X2_batch, X3_batch], training=False) # a list.
                Y_pred_ar_batch = model([X0_batch, X1_batch, X2_batch, rnn_st_batch], training=False) # a list.


                if(t%5==0):
                    frame = RGBs[0, t]

                                                                                        
                    plot_outs([y.numpy() for y in Y_batch], frame, dir_name=f'output/val/{i}/true', file_name=f'{t}.png')
                    plot_outs([y.numpy() for y in Y_pred_spv_batch], frame, dir_name=f'output/val/{i}/pred-spv', file_name=f'{t}.png')   
                    plot_outs([y.numpy() for y in Y_pred_ar_batch], frame, dir_name=f'output/val/{i}/pred-ar', file_name=f'{t}.png')                             
                 



                Y_pred_spv_batch = tf.concat(Y_pred_spv_batch, axis=-1)
                Y_pred_ar_batch = tf.concat(Y_pred_ar_batch, axis=-1)
                Y_batch = tf.concat(Y_batch, axis=-1)
    
                rnn_st_batch = Y_pred_ar_batch[:, STATE_IDX:]
                
                loss_spv = train_loss_fn(Y_batch, Y_pred_spv_batch)
                loss_ar = train_loss_fn(Y_batch[:, :STATE_IDX], Y_pred_ar_batch[:, :STATE_IDX])

                metric_spv = tf.reduce_mean(maxae(Y_batch, Y_pred_spv_batch))
                metric_ar = tf.reduce_mean(maxae(Y_batch[:, :STATE_IDX], Y_pred_ar_batch[:, :STATE_IDX]))

            
                losses_ar.append(loss_ar.numpy())
                losses_spv.append(loss_spv.numpy())
                
                metrics_ar.append(metric_ar.numpy())
                metrics_spv.append(metric_spv.numpy())


            del X0, X3, Y, RGBs
        
        except:
            pass

    return np.mean(losses_ar), np.mean(losses_spv), np.mean(metrics_ar), np.mean(metrics_spv)







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



 

def maxae(y_true, y_pred):
  # [for i in range(len(y_true))]
  return KB.max(KB.abs(y_pred - y_true), axis=-1)

  
 

PATH_DISTANCE = 192
LANE_OFFSET = 1.8
LEAD_X_SCALE = 10   # x_scale in driving.cc
LEAD_Y_SCALE = 10   # y_scale in driving.cc
LEAD_V_SCALE = 1

TIME_DISTANCE = 100
 
PATH_IDX   = 0      # o0:  192*2+1 = 385
LL_IDX     = 385    # o1:  192*2+2 = 386
RL_IDX     = 771    # o2:  192*2+2 = 386
LEAD_IDX   = 1157   # o3:  11*5+3 = 58
LONG_X_IDX = 1215   # o4:  100*2 = 200
LONG_V_IDX = 1415   # o5:  100*2 = 200
LONG_A_IDX = 1615   # o6:  100*2 = 200
DESIRE_IDX = 1815   # o7:  8
META_IDX   = 1823   # o8:  4
PRED_IDX   = 1827   # o9:  32
POSE_IDX   = 1859   # o10: 12
STATE_IDX  = 1871   # o11: 512
OUTPUT_IDX = 2383
 

 


def standardize(y_true, y_pred):

    eps = 1

    mean = tf.math.reduce_mean(y_true)
    std = tf.math.reduce_std(y_true) 

    # print(f'std: {std}')
    std = tf.math.maximum(std, eps)


    return (y_true - mean) / std, (y_pred - mean) / std



def path_loss_fn(path_true, path_pred):
    
    def parse(p):

        path = p[:, :PATH_DISTANCE] 
        # path_stds = tf.math.softplus(p[:, PATH_DISTANCE:PATH_DISTANCE*2])    
        path_stds = p[:, PATH_DISTANCE:PATH_DISTANCE*2]
        # path_valid_len = tf.clip_by_value(p[:, PATH_DISTANCE*2], 5, PATH_DISTANCE)     
        path_valid_len = p[:, PATH_DISTANCE*2]

        return path, path_stds, path_valid_len# * 0.01

    def loss_fn(y_true, y_pred):
        # y_true, y_pred = standardize(y_true, y_pred)
        loss = tf.keras.losses.mse(y_true, y_pred)
        return tf.math.reduce_mean(loss)

    def loss_within_valid_len(p_true, p_pred, l): 
        p_true = p_true[:l] # 1D vector.
        p_pred = p_pred[:l] # 1D vector.
        # p_true, p_pred = standardize(p_true, p_pred)        
        loss = tf.keras.losses.mse(p_true, p_pred)
        return loss  # a scalar.



    path_true, path_std_true, path_valid_len_true = parse(path_true)
    path_pred, path_std_pred, path_valid_len_pred = parse(path_pred)


    # avg_scales['path_true'].append(tf.math.reduce_mean(path_true).numpy())
    # avg_scales['path_std_true'].append(tf.math.reduce_mean(path_std_true).numpy())
    # avg_scales['path_valid_len_true'].append(tf.math.reduce_mean(path_valid_len_true).numpy())

    # L = tf.cast(path_valid_len_true, dtype=tf.int32)
 
    
    # loss_path = tf.map_fn(lambda x: loss_within_valid_len(x[0], x[1], x[2]), (path_true, path_pred, L), dtype=tf.float32) # (b,)
    # loss_std = tf.map_fn(lambda x: loss_within_valid_len(x[0], x[1], x[2]), (path_std_true, path_std_pred, L), dtype=tf.float32) # (b,)

    # print(f'tf.reduce_mean(loss_path): {tf.reduce_mean(loss_path)}, L: {L}')
    # print(f'tf.reduce_mean(loss_std): {tf.reduce_mean(loss_std)}')


    # return tf.reduce_mean(loss_path) + tf.reduce_mean(loss_std) + loss_fn(path_valid_len_true * 0.01, path_valid_len_pred * 0.01)

    return 0.33*loss_fn(path_true, path_pred) + 0.33*loss_fn(path_std_true, path_std_pred) + \
                                    0.33*loss_fn(path_valid_len_true* 0.01, path_valid_len_pred* 0.01)




def lane_loss_fn(lane_true, lane_pred, sign):
    '''
        sign: +1 for left lane, -1 for right lane.
    '''
 
    def parse(l):

        lane = l[:, :PATH_DISTANCE] + sign * LANE_OFFSET 
        # lane_std = tf.math.softplus(l[:, PATH_DISTANCE:PATH_DISTANCE*2])
        lane_std = l[:, PATH_DISTANCE:PATH_DISTANCE*2]
        # lane_valid_len = tf.clip_by_value(l[:, PATH_DISTANCE*2], 5, PATH_DISTANCE) 
        lane_valid_len = l[:, PATH_DISTANCE*2]
        # lane_prob = tf.math.sigmoid(l[:, PATH_DISTANCE*2 + 1])
        lane_prob = l[:, PATH_DISTANCE*2 + 1]

        return lane, lane_std, lane_valid_len, lane_prob


    def loss_fn(y_true, y_pred):
        # y_true, y_pred = standardize(y_true, y_pred)
        loss = tf.keras.losses.mse(y_true, y_pred)
        return tf.math.reduce_mean(loss)

    def loss_within_valid_len(p_true, p_pred, l): 
        p_true = p_true[:l] # 1D vector.
        p_pred = p_pred[:l] # 1D vector.
        # p_true, p_pred = standardize(p_true, p_pred)        
        loss = tf.keras.losses.mse(p_true, p_pred)
        return loss # a scalar.


    lane_true, lane_std_true, lane_valid_len_true, lane_prob_true = parse(lane_true)
    lane_pred, lane_std_pred, lane_valid_len_pred, lane_prob_pred = parse(lane_pred)
    
 
    # avg_scales['lane_true'].append(tf.math.reduce_mean(lane_true).numpy())
    # avg_scales['lane_std_true'].append(tf.math.reduce_mean(lane_std_true).numpy())
    # avg_scales['lane_valid_len_true'].append(tf.math.reduce_mean(lane_valid_len_true).numpy())
    # avg_scales['lane_prob_true'].append(tf.math.reduce_mean(lane_prob_true).numpy())



    # L = tf.cast(lane_valid_len_true, dtype=tf.int32)


    # loss_lane = tf.map_fn(lambda x: loss_within_valid_len(x[0], x[1], x[2]), (lane_true, lane_pred, L), dtype=tf.float32) # (b,)
    # loss_std = tf.map_fn(lambda x: loss_within_valid_len(x[0], x[1], x[2]), (lane_std_true, lane_std_pred, L), dtype=tf.float32) # (b,)

    # return tf.reduce_mean(loss_lane) + tf.reduce_mean(loss_std) + loss_fn(lane_valid_len_true * 0.01, lane_valid_len_pred * 0.01) + loss_fn(lane_prob_true, lane_prob_pred)

    return 0.25*loss_fn(lane_true, lane_pred) + 0.25*loss_fn(lane_std_true, lane_std_pred) + \
            0.25*loss_fn(lane_valid_len_true* 0.01, lane_valid_len_pred* 0.01) + 0.25*loss_fn(lane_prob_true, lane_prob_pred)



 


def lead_loss_fn(lead_true, lead_pred):
  
 
    def parse(l):

        # lead_prob = tf.math.sigmoid(l[:, -3]) # lead.shape = (b, 58)
        
        lead = tf.reshape(l[:, :-3], (-1, 5, 11)) # (b, 5, 11)
 
        lead_weights = tf.nn.softmax(lead[:, :, 8]) # lead_weights.shape = (b, 5)
  
        lidx = tf.math.argmax(lead_weights, axis=-1, output_type=tf.dtypes.int32)   # (b,)         
 
        idxes = tf.stack([tf.range(tf.shape(lidx)[0]), lidx], axis=1)
 
        lead_max = tf.gather_nd(lead, idxes) # (b, 11)
 

        x = lead_max[:, 0] #* LEAD_X_SCALE
        y = lead_max[:, 1] #* LEAD_Y_SCALE
        v = lead_max[:, 2] #* LEAD_V_SCALE
        a = lead_max[:, 3] 

        # x_std = tf.math.softplus(lead_max[:, 4]) * LEAD_X_SCALE
        # y_std = tf.math.softplus(lead_max[:, 5]) * LEAD_Y_SCALE
        # v_std = tf.math.softplus(lead_max[:, 6]) * LEAD_V_SCALE
        # a_std = tf.math.softplus(lead_max[:, 7])

        x_std = lead_max[:, 4] 
        y_std = lead_max[:, 5] 
        v_std = lead_max[:, 6] 
        a_std = lead_max[:, 7]

        # return x, y*100.0, v, a*10.0, x_std, y_std, v_std, a_std, lead_weights
        return x, y, v, a, x_std, y_std, v_std, a_std, lead_weights


    def loss_fn(y_true, y_pred):
        # y_true, y_pred = standardize(y_true, y_pred)
        loss = tf.keras.losses.mse(y_true, y_pred)
        return tf.math.reduce_mean(loss)

    x_true, y_true, v_true, a_true, x_std_true, y_std_true, v_std_true, a_std_true, lead_weights_true = parse(lead_true)
    x_pred, y_pred, v_pred, a_pred, x_std_pred, y_std_pred, v_std_pred, a_std_pred, lead_weights_pred = parse(lead_pred)
    
   

    # avg_scales['x_true'].append(tf.math.reduce_mean(x_true).numpy())
    # avg_scales['y_true'].append(tf.math.reduce_mean(y_true).numpy())
    # avg_scales['v_true'].append(tf.math.reduce_mean(v_true).numpy())
    # avg_scales['a_true'].append(tf.math.reduce_mean(a_true).numpy())
    # avg_scales['x_std_true'].append(tf.math.reduce_mean(x_std_true).numpy())
    # avg_scales['y_std_true'].append(tf.math.reduce_mean(y_std_true).numpy())
    # avg_scales['v_std_true'].append(tf.math.reduce_mean(v_std_true).numpy())
    # avg_scales['a_std_true'].append(tf.math.reduce_mean(a_std_true).numpy())
    # avg_scales['lead_weights_true'].append(tf.math.reduce_mean(lead_weights_true).numpy())

    return 0.11*loss_fn(x_true, x_pred) + 0.11*loss_fn(y_true, y_pred) + 0.11*loss_fn(v_true, v_pred) + \
           0.11*loss_fn(a_true, a_pred) + 0.11*loss_fn(x_std_true, x_std_pred) + 0.11*loss_fn(y_std_true, y_std_pred) + \
            0.11*loss_fn(v_std_true, v_std_pred) + 0.11*loss_fn(a_std_true, a_std_pred) + 0.11*loss_fn(lead_weights_true, lead_weights_pred)

 


def longi_loss_fn(longi_true, longi_pred):
   
 
    loss = tf.math.square(longi_true[:, :TIME_DISTANCE][:, ::10] - longi_pred[:, :TIME_DISTANCE][:, ::10])

    loss = tf.math.reduce_mean(loss, axis=-1)

    return tf.math.reduce_mean(loss)

 




def train_loss_fn(y_true, y_pred): 

    path_loss = path_loss_fn(y_true[:, PATH_IDX:LL_IDX], y_pred[:, PATH_IDX:LL_IDX])
    ll_loss = lane_loss_fn(y_true[:, LL_IDX:RL_IDX], y_pred[:, LL_IDX:RL_IDX], +1)
    rl_loss = lane_loss_fn(y_true[:, RL_IDX:LEAD_IDX], y_pred[:, RL_IDX:LEAD_IDX], -1)
    lead_loss = lead_loss_fn(y_true[:, LEAD_IDX:LONG_X_IDX], y_pred[:, LEAD_IDX:LONG_X_IDX])
 
   
    longi_x_loss = longi_loss_fn(y_true[:, LONG_X_IDX:LONG_V_IDX], y_pred[:, LONG_X_IDX:LONG_V_IDX])
    longi_v_loss = longi_loss_fn(y_true[:, LONG_V_IDX:LONG_A_IDX], y_pred[:, LONG_V_IDX:LONG_A_IDX])
    longi_a_loss = longi_loss_fn(y_true[:, LONG_A_IDX:DESIRE_IDX], y_pred[:, LONG_A_IDX:DESIRE_IDX])

    
    return 0.15*path_loss + 0.15*ll_loss + 0.15*rl_loss + 0.15*lead_loss + 0.1*tf.math.reduce_mean(tf.keras.losses.mse(y_true[:, STATE_IDX:], y_pred[:, STATE_IDX:])) + 0.1*longi_x_loss + 0.1*longi_v_loss + 0.1*longi_a_loss

    # loss_CS = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
    # loss_MSE = tf.keras.losses.mse(y_true, y_pred)
    # total_loss = 0.5 * loss_CS + 0.5 * loss_MSE  # MC5
    # return loss
        
    # return /tf.math.reduce_mean(total_loss)

    

 

model = get_model()


if 'ckpt_load_path' in para:
    model.load_weights(para.ckpt_load_path)  # for retraining
  
optimizer = tf.keras.optimizers.AdamW(learning_rate=para.base_lr,
                  weight_decay=para.weight_decay, ema_momentum=para.ema_momentum, clipvalue=1.0)        

 
all_pkl = glob.glob("/home/richard/Downloads/TData1/*.pkl") 
split = int(len(all_pkl) * 0.85)
train_pkl = all_pkl[:split]
val_pkl = all_pkl[split:]
 


train_episodes = gen_episodes_train(train_pkl)
val_episodes = gen_episodes_val(val_pkl)

 

# with open("log.txt", "w") as f: f.write("")
# with open("val.txt", "w") as f: f.write("")



B = para.batch_size
H = para.horizon
# U = para.update_interval


  
 

@tf.function
def train_step(X_step, Y_step):
  
    
    Y_step = tf.concat(Y_step, axis=-1) 

  
    with tf.GradientTape() as tape:
           
       
        Y_pred = model(X_step, training=True)  
        Y_pred = tf.concat(Y_pred, axis=-1) 
 
        loss = train_loss_fn(Y_step, Y_pred)   
 
    grad = tape.gradient(loss, model.trainable_variables)

    metric = tf.reduce_mean(maxae(Y_step, Y_pred))

    return loss, metric, grad






# try:
#     print(f'validating...') 
#     loss_ar, loss_spv, metric_ar, metric_spv = validate(32)
#     log = f'[{update_counts}] val loss_ar: {loss_ar}, loss_spv: {loss_spv}, metric_ar: {metric_ar}, metric_spv: {metric_spv}'
#     print(log)
#     with open("val.txt", "a") as f: f.write(log + '\n')                
# except:
#     pass



# exit()


# with open("train.txt", "w") as f: f.write("")
# with open("val.txt", "w") as f: f.write("")

 
update_counts = 0

for epoch in range(100000):


    for epid, episodes in enumerate(train_episodes):  
  
        X0, X3, Y = episodes                

        for t in range(H):

            accum_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
            step_size = para.batch_size // para.accum_steps            
            
            X1_step = tf.convert_to_tensor(np.zeros((step_size, 8)), dtype=tf.float32)
            X2_step = tf.convert_to_tensor(np.zeros((step_size, 2)), dtype=tf.float32)            

            total_loss = 0.0
            total_metric = 0.0

            for step in range(para.accum_steps): 
    
                X0_step = tf.convert_to_tensor(X0[step*step_size:(step+1)*step_size, t, ...], dtype=tf.float32)
                X3_step = tf.convert_to_tensor(X3[step*step_size:(step+1)*step_size, t, ...], dtype=tf.float32)
                Y_step = [tf.convert_to_tensor(y[step*step_size:(step+1)*step_size, t, ...], dtype=tf.float32) for y in Y]
             
                X_step = [X0_step, X1_step, X2_step, X3_step]

                loss, metric, grad = train_step(X_step, Y_step)
    
    
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
                log = f'[{update_counts}] train loss: {total_loss / para.accum_steps}' + f', metric: {total_metric / para.accum_steps}' + f', lr: {lr_next}' 
                print(log)
                with open("train.txt", "a") as f: f.write(log + '\n')

            

            if update_counts % para.validate_interval == 0:
                try:
                    print(f'validating...') 
                    loss_ar, loss_spv, metric_ar, metric_spv = validate()
                    log = f'[{update_counts}] val loss_ar: {loss_ar}, loss_spv: {loss_spv}, metric_ar: {metric_ar}, metric_spv: {metric_spv}'
                    print(log)
                    with open("val.txt", "a") as f: f.write(log + '\n')                
                except:
                    pass

            if update_counts % para.save_interval == 0:
                model.save_weights(f'ckpt/modelB6-{update_counts}.h5')
                print(f'saved ckpt: ckpt/modelB6-{update_counts}.h5')





 