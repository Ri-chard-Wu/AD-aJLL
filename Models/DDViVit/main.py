import os
import sys
import time
import random
from tqdm import tqdm
from argparse import ArgumentParser

import matplotlib.pyplot as plt
 
import numpy as np
import tensorflow as tf

from model import MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
import cv2
import glob

import pickle
from cameraB3 import transform_img, eon_intrinsics, medmodel_intrinsics, draw_path
 
 
from parameter import train_args as args
from parameter import transformerEncoder_args as enc_args

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
 


Y_shape = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]


 

 
 
def RGB_to_sYUVs(frame):
   
    bYUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420) # (1311, 1164)

    sYUV = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
                            yuv=True, output_size=(512, 256))  # (384, 512)
    
    return sYUV

         

def sYUVs_to_CsYUVs(sYUVs): # sYUVs: (384, 512)

    
    H = (sYUVs.shape[0]*2)//3  # 384x2//3 = 256
    W = sYUVs.shape[1]         # 512
    CsYUVs = np.zeros((6, H//2, W//2), dtype=np.uint8)

    CsYUVs[0] = sYUVs[0:H:2, 0::2]  # [2::2] get every even starting at 2
    CsYUVs[1] = sYUVs[1:H:2, 0::2]  # [start:end:step], [2:4:2] get every even starting at 2 and ending at 4
    CsYUVs[2] = sYUVs[0:H:2, 1::2]  # [1::2] get every odd index, [::2] get every even
    CsYUVs[3] = sYUVs[1:H:2, 1::2]  # [::n] get every n-th item in the entire sequence
    CsYUVs[4] = sYUVs[H:H+H//4].reshape((H//2, W//2))
    CsYUVs[5] = sYUVs[H+H//4:H+H//2].reshape((H//2, W//2))

 
    return CsYUVs # (6, 128, 256)


def RGB_to_YUV(frame):

    if frame is None: return None

    frame = RGB_to_sYUVs(frame)
    frame = sYUVs_to_CsYUVs(frame)
    return frame


def read_frames(hevc_file):

    frames = []

    cap = cv2.VideoCapture(hevc_file)

    ret, frame = cap.read()  # frame: (874, 1164, 3) bgr img in uint8 np array.
     
    while ret:        
        frames.append(frame)
        ret, frame = cap.read()  

    cap.release()
    return frames
 
  
 
 



def get_train_dataloader(pkl_files):
  
    hevc_files = [pkl_file.replace('data.pkl', 'fcamera.hevc') for pkl_file in pkl_files]    

    # H = args.horizon 
    H = 256
    B = args.batch_size
    n_B = len(pkl_files) // B 
    seq_len = enc_args.seq_len # 32.
    frame_skip = 4
    
    # seq_skip = seq_len // 2 # 16.
    seq_skip = 1

    fidx = 0
    epoch = 0

    while(1):  
        
        # RGBs = np.zeros((1, H, 874, 1164, 3), dtype=np.uint8) # for debug.
        X0 = np.zeros((B, H, 12, 128, 256), dtype=np.uint8)            
        X3 = np.zeros((B, H, 512), dtype=np.float32)    
        Y = [np.zeros((B, H, s)) for s in Y_shape]

        print(f'epoch{epoch}, file {fidx+1} / {len(pkl_files)}')

        for bidx in range(B):
            
            while(1):
                pkl_file = pkl_files[fidx]
                hevc_file = hevc_files[fidx]                    
                frames = read_frames(hevc_file) 
                frames = frames[::frame_skip]

                fidx += 1
                if(fidx >= len(pkl_files)): 
                    fidx = 0
                    epoch += 1

                if(len(frames) > H):  
                    break
                else:
                    print(f'[Warning] len(frames) too small): {hevc_file}')                        
                    continue
             
            
            with open(pkl_file, 'rb') as f:      

                data = pickle.load(f)   
                for i in range(12):
                    assert data['Y'][i].shape[1] == Y_shape[i]

            for j in range(12):
                data['Y'][j] = data['Y'][j][::frame_skip]
                
            for t in range(H):   
                
                # RGBs[0, t] = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
                X0[bidx, t] = np.vstack((RGB_to_YUV(frames[t]), RGB_to_YUV(frames[t+1])))                
                for j in range(12):
                    Y[j][bidx, t] = data['Y'][j][t]

            print(f'\rsampling training data {bidx+1} / {B}', end="")

        print()

        for t in range(0, H-seq_len, seq_skip): 
            # yield X0[:, t:t+seq_len, :, :, :], Y[0][:, t+seq_len-1, :args.num_pts] # (b, seq_len, 12, 128, 256), (b, num_pts).
            yield X0[:, t:t+seq_len, :, :, :], Y[0][:, t+seq_len-1] # (b, seq_len, 12, 128, 256), (b, num_pts).
            
        del X0, Y, X3, data

 




def get_val_dataloader(pkl_files):
  
    hevc_files = [pkl_file.replace('data.pkl', 'fcamera.hevc') for pkl_file in pkl_files]    

    H = args.horizon_val
    seq_len = enc_args.seq_len
    frame_skip = 4
  
    while(1):

        idx = np.random.choice(np.arange(len(pkl_files)), size=1)[0]
        pkl_file = pkl_files[idx]
        hevc_file = hevc_files[idx]

        RGBs = np.zeros((1, H, 874, 1164, 3), dtype=np.uint8) # for debug.
        X0 = np.zeros((1, H, 12, 128, 256), dtype=np.uint8)            
        Y = [np.zeros((1, H, s)) for s in Y_shape]
     
        frames = read_frames(hevc_file) 

        frames = frames[::frame_skip]


        # if(len(frames) <= H):  
 
        if(len(frames) <= H): 
            print(f'[Warning] len(frames)-H <= 0): {hevc_file}')
            continue

   
 
        with open(pkl_file, 'rb') as f:      

            data = pickle.load(f)    
            
            for i in range(12):
                assert data['Y'][i].shape[1] == Y_shape[i]

        for j in range(12):
            data['Y'][j] = data['Y'][j][::frame_skip]
            
            
        for t in range(H):   

            RGBs[0, t] = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
            X0[0, t] = np.vstack((RGB_to_YUV(frames[t]), RGB_to_YUV(frames[t+1])))

            for j in range(12):
                Y[j][0, t] = data['Y'][j][t]


        skip = 32
        for t in range(0, H-seq_len, skip): 
            yield X0[:, t:t+seq_len, :, :, :], Y[0][:, t+seq_len-1], RGBs[:, t+seq_len-1, :, :, :] # (b, seq_len, 12, 128, 256), (b, 2*num_pts+1), (1, 874, 1164, 3).        

        del X0, Y, RGBs, data

  

def plot_bsv(traj_true, traj_pred, dir_name, file_name):

 
     
    if(not os.path.exists(dir_name)):
        os.makedirs(dir_name, exist_ok=True)

    PATH_DISTANCE = 192
    x_lspace = np.linspace(1, PATH_DISTANCE, PATH_DISTANCE)  
    
    path_true = traj_true[:PATH_DISTANCE]
    # path_std_true = traj_true[PATH_DISTANCE:2*PATH_DISTANCE]
    valid_len_true = np.fmin(PATH_DISTANCE, np.fmax(5, traj_true[2*PATH_DISTANCE]))
 
    path_pred = traj_pred[:PATH_DISTANCE]
    # path_std_pred = traj_pred[PATH_DISTANCE:2*PATH_DISTANCE]
    # valid_len_pred = np.fmin(PATH_DISTANCE, np.fmax(5, traj_pred[2*PATH_DISTANCE]))
    valid_len_pred = traj_pred[2*PATH_DISTANCE]



    plt.clf()   # clear figure
    plt.xlim(0, 1200)
    plt.ylim(800, 0)
 
    l_true = int(valid_len_true)  

    # -----------------------   
    
    plt.subplot(223)
    plt.gca().invert_xaxis() 
    # plt.title("true bsv") 
    plt.plot(path_true[4:l_true], x_lspace[:l_true-4],  linewidth=1, label='true')
    plt.plot(path_pred[4:l_true], x_lspace[:l_true-4],  linewidth=1, label='pred')
    plt.legend()
    # ----------------------- 

    # plt.subplot(224)
    # plt.gca().invert_xaxis() 
    # plt.title("pred bsv") 
    # plt.plot(path_pred[4:l_true], x_lspace[:l_true-4], "g-", linewidth=1)
  
    # ----------------------- 

    plt.tight_layout()
    plt.savefig(dir_name + '/' + file_name)  



 


def plot_outs(traj_true, # (2*num_pts+1,).
            traj_pred, # (2*num_pts+1,).
             frame, 
             dir_name, 
             file_name):

    frame = frame.astype(np.uint8)
 
     
    if(not os.path.exists(dir_name)):
        os.makedirs(dir_name, exist_ok=True)

    PATH_DISTANCE = 192
    x_lspace = np.linspace(1, PATH_DISTANCE, PATH_DISTANCE)  
    
    path_true = traj_true[:PATH_DISTANCE]
    # path_std_true = traj_true[PATH_DISTANCE:2*PATH_DISTANCE]
    valid_len_true = np.fmin(PATH_DISTANCE, np.fmax(5, traj_true[2*PATH_DISTANCE]))
 
    path_pred = traj_pred[:PATH_DISTANCE]
    # path_std_pred = traj_pred[PATH_DISTANCE:2*PATH_DISTANCE]
    # valid_len_pred = np.fmin(PATH_DISTANCE, np.fmax(5, traj_pred[2*PATH_DISTANCE]))
    valid_len_pred = traj_pred[2*PATH_DISTANCE]



    plt.clf()   # clear figure
    plt.xlim(0, 1200)
    plt.ylim(800, 0)

    # ----------------------- 
    plt.subplot(221) # 221: 2 rows, 2 columns, 1st sub-figure 
    l_true = int(valid_len_true)    
    plt.imshow(draw_path(frame.copy(), path_true[4:l_true], x_lspace[:l_true-4])) 
    plt.title(f"true, l: {l_true}")

    # ----------------------- 
    plt.subplot(222)   
    # l_pred = int(valid_len_pred)    
    plt.imshow(draw_path(frame.copy(), path_pred[4:l_true], x_lspace[:l_true-4]))  
    plt.title(f"pred, l: {valid_len_pred}")

    # ----------------------- 
 
    plt.subplot(223)
    plt.gca().invert_xaxis() 
    plt.title("true bsv") 
    plt.plot(path_true[4:l_true], x_lspace[:l_true-4], "g-", linewidth=1)
 
    # ----------------------- 

    plt.subplot(224)
    plt.gca().invert_xaxis() 
    plt.title("pred bsv") 
    plt.plot(path_pred[4:l_true], x_lspace[:l_true-4], "g-", linewidth=1)
  
    # ----------------------- 

    plt.tight_layout()
    plt.savefig(dir_name + '/' + file_name)  



 


def validate(n=16):
  
    seq_len = enc_args.seq_len
        
    for i, data in enumerate(val_dataloader):

        if i >= n: break
 

        inputs, labels, RGBs = data # (b, seq_len, 12, 128, 256), (1, 2*num_pts+1), (1, 874, 1164, 3).

        inputs_t = tf.convert_to_tensor(inputs, dtype=tf.float32) # (b, seq_len, 12, 128, 256).
        # labels = tf.convert_to_tensor(labels, dtype=tf.float32) # (b, 2*num_pts+1).

        input_past = inputs_t[:,:seq_len-1, :, :, :] # (b, seq_len-1, 12, 128, 256).
        input_cur = inputs_t[:,seq_len-1, :, :, :] # (b, 12, 128, 256).

      
        feature_past = tf.map_fn(model.extract_feature, input_past) # (b, seq_len-1, 768).             
 
        traj_pred = model(input_cur, feature_past, training=False)[0] # (2*num_pts+1). 

        
        plot_outs(labels[0], traj_pred, RGBs[0], dir_name=f'output/val', file_name=f'{i}.png')
     
        


        del labels, inputs, RGBs

        print(f'\rvalidating {i+1} / {n}', end="")

    print()    



 
def lr_scheduler(step):

    step = min(step, args.total_steps)

    lr = args.base_lr

    progress = (step - args.warmup_steps) / float(args.total_steps - args.warmup_steps)
    progress = np.clip(progress, 0.0, 1.0)
  
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    
    # if args.warmup_steps:
    lr = lr * np.minimum(1., step / args.warmup_steps)

    lr = max(lr, args.lr_min)

    return np.asarray(lr, dtype=np.float32) 



model = SequencePlanningNetwork()
model(tf.random.uniform((1, 12, 128, 256)), tf.random.uniform((1, enc_args.seq_len-1, enc_args.hidden_size)))


if args.ckpt: 
    model.load_weights(args.ckpt)  # for retraining
    print(f'loaded ckpt: {args.ckpt}')





# model.summary()
# exit()
# optimizer, lr_scheduler = model.configure_optimizers(args, model)

optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr_min,
                weight_decay=0.01, ema_momentum=0.9, clipvalue=1.0)        


# if args.resume and rank == 0:
#     print('Loading weights from', args.resume)
#     model.load_state_dict(torch.load(args.resume), strict=True)


loss_fn = MultipleTrajectoryPredictionLoss()


all_pkl = glob.glob("/home/richard/Downloads/TData1/*.pkl") 
split = int(len(all_pkl) * 0.85)
train_pkl = all_pkl[:split]
val_pkl = all_pkl[split:]

train_dataloader = get_train_dataloader(train_pkl)
val_dataloader = get_val_dataloader(val_pkl)


bs = args.batch_size
eps_len = args.horizon 
seq_len = enc_args.seq_len
n_seq = eps_len//seq_len
bstep = bs//args.accum_steps

 
for epid, data in enumerate(train_dataloader):   

    inputs, labels = data # (b, seq_len, 12, 128, 256), (b, 2*num_pts+1). 
        
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32) # (b, seq_len, 12, 128, 256).
    labels = tf.convert_to_tensor(labels, dtype=tf.float32) # (b, 2*num_pts+1).

    input_past = inputs[:,:seq_len-1, :, :, :] # (b, seq_len-1, 12, 128, 256).
    input_cur = inputs[:,seq_len-1, :, :, :] # (b, 12, 128, 256).

    accum_gradients = [tf.zeros_like(var) for var in model.trainable_variables]


    for aidx in range(args.accum_steps):
        
        input_past_mb = input_past[aidx*bstep:(aidx+1)*bstep, ...]
        input_cur_mb = input_cur[aidx*bstep:(aidx+1)*bstep, ...]
        labels_mb = labels[aidx*bstep:(aidx+1)*bstep, ...]

        # feature_past = tf.map_fn(model.extract_feature, input_past_mb) # (b, seq_len-1, 768).             
        feature_past = tf.zeros((bstep, seq_len-1, enc_args.hidden_size))

        with tf.GradientTape() as tape:  
            traj_pred = model(input_cur_mb, feature_past) # (b, 2*num_pts+1).  
            path_loss, valid_len_loss, std_loss = loss_fn(traj_pred, labels_mb) # (,), (,).
            loss = path_loss + 0.1*valid_len_loss + 0.1*std_loss
        
        grad = tape.gradient(loss, model.trainable_variables)


        for i in range(len(accum_gradients)):
            accum_gradients[i] += grad[i] 

    
    for i in range(labels_mb.shape[0]):
        plot_bsv(labels_mb[i], traj_pred[i], dir_name=f'output/train', file_name=f'tr-{i}.png')


    averaged_gradients = [accum_grad / tf.cast(args.accum_steps, tf.float32) for accum_grad in accum_gradients]

    # print(f'averaged_gradients: {averaged_gradients}')
    optimizer.apply_gradients(zip(averaged_gradients, model.trainable_variables))


    lr_next = lr_scheduler(epid)
    tf.keras.backend.set_value(optimizer.learning_rate, lr_next)


    if (epid+1) % args.log_interval == 0:
        log = f'[{epid}] path_loss: {round(float(path_loss.numpy()), 3)}, valid_len_loss: {round(float(valid_len_loss.numpy()), 3)}, std_loss: {round(float(std_loss.numpy()), 3)}, lr: {lr_next}.'
        print(log)
        with open("train.txt", "a") as f: f.write(log + '\n')
        

    if (epid+1) % args.log_wandb_interval == 0:
        for i in range(len(accum_gradients)): 
            print(f'[grad-{i}] mean: {tf.math.reduce_mean(tf.math.abs(grad[i]))}, max: {tf.math.reduce_max(tf.math.abs(grad[i]))}, min: {tf.math.reduce_min(tf.math.abs(grad[i]))}')
    
    
    if (epid+1) % args.val_interval == 0:
        validate()

    if (epid+1) % args.save_interval == 0:
        model.save_weights(f'ckpt/DDViVit-{epid}.h5')
        print(f'saved ckpt: ckpt/DDViVit-{epid}.h5')




        
    # lr_scheduler.step()



 
