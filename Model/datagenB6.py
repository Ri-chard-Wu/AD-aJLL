"""   YPL, YJW, JLL, 2021.9.8 - 2024.2.8
for 230826, 231229, 240124
from /home/richard/openpilot/aJLLold/Model/datagenB6c.py
Input:
  /models/supercombo079.keras
  /home/richard/dataB6/UHD--2018-08-02--08-34-47--32/yuv.h5
Output:
  Ximgs.shape = (none, 2x6, 128, 256)  (num_channels = 6, 2 yuv images)
  Xin1 = (none, 8)
  Xin2 = (none, 2)
  Xin3 = (none, 512)
  Ytrue0 = outs[ 0 ].shape = (none, 385)
  Ytrue1 = outs[ 1 ].shape = (none, 386)
  Ytrue2 = outs[ 2 ].shape = (none, 386)
  Ytrue3 = outs[ 3 ].shape = (none, 58)
  Ytrue4 = outs[ 4 ].shape = (none, 200)
  Ytrue5 = outs[ 5 ].shape = (none, 200)
  Ytrue6 = outs[ 6 ].shape = (none, 200)
  Ytrue7 = outs[ 7 ].shape = (none, 8)
  Ytrue8 = outs[ 8 ].shape = (none, 4)
  Ytrue9 = outs[ 9 ].shape = (none, 32)
  Ytrue10 = outs[ 10 ].shape = (none, 12)
  Ytrue11 = outs[ 11 ].shape = (none, 512)
"""
import os
import cv2
import h5py
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
 
from tools.lib.framereader import FrameReader
from common.transformations.model import medmodel_intrinsics
from cameraB3 import transform_img, eon_intrinsics
from PIL import Image
import glob
import pickle
import time

 
from tqdm import tqdm


 
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# with tf.device('/CPU:0'):
supercombo = load_model('saved_model/supercombo079.keras', compile=False)  



def supercombo_y(Ximgs, Xin1, Xin2, Xin3):

  # supercombo = load_model('saved_model/supercombo079.keras', compile=False)  

    #--- np.shape(Ximgs) = (12, 128, 256)
  Ximgs = np.expand_dims(Ximgs, axis=0)
    #--- np.shape(Ximgs) = (1, 12, 128, 256)
    # x = np.array([1, 2]) > x.shape: (2,) > y = np.expand_dims(x, axis=0) > y: array([[1, 2]]) > y.shape: (1, 2)
  Xin1 = np.expand_dims(Xin1, axis=0)
  Xin2 = np.expand_dims(Xin2, axis=0)
  Xin3 = np.expand_dims(Xin3, axis=0)
  inputs = [Ximgs, Xin1, Xin2, Xin3]

  # with tf.device('/CPU:0'):
  outputs = supercombo(inputs)

    #[print("#--- outputs[", i, "].shape =", np.shape(outputs[i])) for i in range(len(outputs))]
  return outputs




def sYUV_to_CsYUV(sYUV):
  H = (sYUV.shape[0]*2)//3  # 384x2//3 = 256
  W = sYUV.shape[1]         # 512
  CsYUV = np.zeros((6, H//2, W//2), dtype=np.uint8)

  CsYUV[0] = sYUV[0:H:2, 0::2]  # [2::2] get every even starting at 2
  CsYUV[1] = sYUV[1:H:2, 0::2]  # [start:end:step], [2:4:2] get every even starting at 2 and ending at 4
  CsYUV[2] = sYUV[0:H:2, 1::2]  # [1::2] get every odd index, [::2] get every even
  CsYUV[3] = sYUV[1:H:2, 1::2]  # [::n] get every n-th item in the entire sequence
  CsYUV[4] = sYUV[H:H+H//4].reshape((-1, H//2, W//2))
  CsYUV[5] = sYUV[H+H//4:H+H//2].reshape((-1, H//2, W//2))

  CsYUV = np.array(CsYUV).astype(np.float32)
  return CsYUV






from common.transformations.model import medmodel_intrinsics
from lanes_image_space import transform_points
from cameraB3 import transform_img, eon_intrinsics
from parserB6 import parser

def plot_path(outs, frame, i):

    PATH_DISTANCE = 192
    x_lspace = np.linspace(1, PATH_DISTANCE, PATH_DISTANCE)  
    
    outs = [a.numpy() for a in outs]
 
    parsed = parser(outs)
      #--- len(parsed) = 25
      #[print("#--- parsed[", x, "].shape =", parsed[x].shape) for x in parsed]   # see output.txt
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # cv2 reads images in BGR format (instead of RGB)

    plt.clf()   # clear figure
    plt.xlim(0, 1200)
    plt.ylim(800, 0)

    plt.subplot(221)   # 221: 2 rows, 2 columns, 1st sub-figure
    plt.title("Overlay Scene")
      # lll = left lane line, path = path line, rll = right lane line
    new_x_left, new_y_left = transform_points(x_lspace, parsed["lll"][0])
    new_x_path, new_y_path = transform_points(x_lspace, parsed["path"][0])
    new_x_right, new_y_right = transform_points(x_lspace, parsed["rll"][0])

    plt.plot(new_x_left, new_y_left, label='transformed', color='r')
    plt.plot(new_x_path, new_y_path, label='transformed', color='g')
    plt.plot(new_x_right, new_y_right, label='transformed', color='b')

    plt.imshow(frame)   # Merge raw image and plot together

    plt.subplot(222)
    plt.gca().invert_yaxis()
    plt.title("Camera View")
    # new_x_left, new_y_left = transform_points(x_lspace, parsed["lll"][0])
    # new_x_path, new_y_path = transform_points(x_lspace, parsed["path"][0])
    # new_x_right, new_y_right = transform_points(x_lspace, parsed["rll"][0])

    plt.plot(new_x_left, new_y_left, label='transformed', color='r')
    plt.plot(new_x_path, new_y_path, label='transformed', color='g')
    plt.plot(new_x_right, new_y_right, label='transformed', color='b')

    plt.legend(['left', 'path', 'right'])

    plt.subplot(223)
    plt.title("Original Scene")
    plt.imshow(frame)

    plt.subplot(224)
    plt.gca().invert_xaxis()
      # Needed to invert axis because standard left lane is positive and right lane is negative, so we flip the x axis
    plt.title("Top-Down View")
    plt.plot(parsed["lll"][0], range(0, PATH_DISTANCE), "r-", linewidth=1)
    plt.plot(parsed["path"][0], range(0, PATH_DISTANCE), "g-", linewidth=1)
    plt.plot(parsed["rll"][0], range(0, PATH_DISTANCE), "b-", linewidth=1)

    
    plt.savefig(f'output/gen/gen-{i}.png')  



def read_hevc(hevc_file):

    frames_rgb = []

    cap = cv2.VideoCapture(hevc_file)

    ret, frame = cap.read()  # bgr img.
    

    count = 0
    while ret: 
        count += 1   

        # frame = Image.fromarray(frame)
        # frame = frame.resize((256, 256), Image.BILINEAR) 
        # frame = np.array(frame) # (84, 84, 3)

        # frame = (frame - 127.5) / 127.5
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # rgb img.
        frames_rgb.append(frame)

        ret, frame = cap.read() 
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # rgb img.
    
    return frames_rgb


          
# def datagen(batch_size, camera_files):
#   print('################################')
#   # 2 yub imgs.
#   Ximgs  = np.zeros((batch_size, 12, 128, 256), dtype='float32')   # Is YUV input img uint8? No. See float8 ysf = convert_float8(ys) in loadyuv.cl

#   # Ximgs_rgb  = np.zeros((batch_size, 256, 256, 3), dtype='float32') 


#   Xin1   = np.zeros((batch_size, 8), dtype='float32')     # DESIRE_LEN = 8
#   Xin2   = np.zeros((batch_size, 2), dtype='float32')     # TRAFFIC_CONVENTION_LEN = 2
#   Xin3   = np.zeros((batch_size, 512), dtype='float32')   # rnn state
  
#   Y_shapes = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]
#   Y = [np.zeros((batch_size, s), dtype='float32') for s in Y_shapes]

#   # Ytrue0 = np.zeros((batch_size, 385), dtype='float32')
#   # Ytrue1 = np.zeros((batch_size, 386), dtype='float32')
#   # Ytrue2 = np.zeros((batch_size, 386), dtype='float32')
#   # Ytrue3 = np.zeros((batch_size, 58), dtype='float32')
#   # Ytrue4 = np.zeros((batch_size, 200), dtype='float32')
#   # Ytrue5 = np.zeros((batch_size, 200), dtype='float32')
#   # Ytrue6 = np.zeros((batch_size, 200), dtype='float32')
#   # Ytrue7 = np.zeros((batch_size, 8), dtype='float32')
#   # Ytrue8 = np.zeros((batch_size, 4), dtype='float32')
#   # Ytrue9 = np.zeros((batch_size, 32), dtype='float32')
#   # Ytrue10 = np.zeros((batch_size, 12), dtype='float32')
#   # Ytrue11 = np.zeros((batch_size, 512), dtype='float32')

#   # Xin1[:, 0] = 1.0   # go straight? desire_state_prob[0] = 1.0
#   # Xin2[:, 0] = 1.0   # traffic_convention[0] = 1.0 = left hand drive like in Taiwan


  
#   hevc_files = glob.glob("/home/richard/dataB6/*/video.hevc")
#   yuv_files = glob.glob("/home/richard/dataB6/*/yuv.h5")
  
  

#   print(f'hevc_files: {hevc_files}')
#   print(f'yuv_files: {yuv_files}')



#   Epoch = 1
#   Xin3_temp = Xin3[0]  #--- np.shape(Xin3_temp) = (512,)

 
#   while True:

#       CFile = 1
#       Step = 1
 
#       assert len(yuv_files) == 3

#       for i in range(len(yuv_files)):

#           yuv_file = yuv_files[i]
#           hevc_file = hevc_files[i]

#           frames_rgb = read_hevc(hevc_file)


#           with h5py.File(yuv_file, "r") as yuv:     

#               yuvX = yuv['X'] #--- yuvX.shape = (1200, 6, 128, 256) or (1199, 6, 128, 256)               

#               yuvX_len = len(yuvX)  # = 1200 or 1199
#               lastIdx = yuvX_len - 2 - batch_size   # cannot be the last frame yuvX_len-1

#               for i in range(0, lastIdx, batch_size):

#                   print("#--- Epoch:", Epoch, " Step:", Step, " CFile:", CFile)

#                   bcount = 0

#                   while bcount < batch_size:
                      
#                       vsX1 = yuvX[bcount+i]
#                       vsX2 = yuvX[bcount+i+1]

#                       # rgb = frames_rgb[bcount+i]

#                         #--- vsX2.shape = (6, 128, 256)
#                       Ximgs[bcount] = np.vstack((vsX1, vsX2))  
#                         #--- Ximgs[bcount].shape = (12, 128, 256)
                      
#                       # Ximgs_rgb[bcount] = frames_rgb[bcount+i]

#                       Xin3[bcount] = Xin3_temp


#                       # start_time = time.time()
#                       outs = supercombo_y(Ximgs[bcount], Xin1[bcount], Xin2[bcount], Xin3[bcount])
#                       # print(f'dt: {time.time() - start_time} sec')
                       
#                       # print(f'outs.shape: {outs.shape}')

#                       if((bcount+i) % 10 == 0):
#                           plot_path(outs, frames_rgb[bcount+i], bcount+i)

#                         #--- len(outs) = 12
#                       Xin3_temp = outs[11][0]
#                         #--- np.shape(outs[11][0]) = (512,)  np.shape(outs[11]) = (1, 512)

#                       Ytrue0[bcount] = outs[0]
#                       Ytrue1[bcount] = outs[1]
#                       Ytrue2[bcount] = outs[2]
#                       Ytrue3[bcount] = outs[3]
#                       Ytrue4[bcount] = outs[4]
#                       Ytrue5[bcount] = outs[5]
#                       Ytrue6[bcount] = outs[6]
#                       Ytrue7[bcount] = outs[7]
#                       Ytrue8[bcount] = outs[8]
#                       Ytrue9[bcount] = outs[9]
#                       Ytrue10[bcount] = outs[10]
#                       Ytrue11[bcount] = outs[11]

#                       assert len(outs) == 12

#                       for i in len(outs):
#                           Y[i][bcount] = outs[i]

#                       bcount += 1

#                   yield Ximgs, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, \
#                             Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11
                  
#                   Step += 1

#           CFile += 1

#       Epoch += 1





          
def datagen(batch_size=None, camera_files=None):
 
    
    hevc_files = glob.glob("/home/richard/dataB6/*/video.hevc")
    yuv_files = glob.glob("/home/richard/dataB6/*/yuv.h5")
    

    print(f'hevc_files: {hevc_files}')
    print(f'yuv_files: {yuv_files}')



    assert len(yuv_files) == 3

    for i in range(len(yuv_files)):

        yuv_file = yuv_files[i]
        hevc_file = hevc_files[i]

        pkl_file = yuv_file.replace('yuv.h5', 'data.pkl')
        # print(f'pkl_file: {pkl_file}')

        with h5py.File(yuv_file, "r") as yuv:     

            yuvX_len = yuv['X'].shape[0] 
            yuvX = yuv['X'] #--- yuvX.shape = (1200, 6, 128, 256) or (1199, 6, 128, 256)               


            # yuvX_len = 10




            Ximgs  = np.zeros((yuvX_len-1, 12, 128, 256), dtype='float32')    
            Xin1   = np.zeros((yuvX_len-1, 8), dtype='float32')     # DESIRE_LEN = 8
            Xin2   = np.zeros((yuvX_len-1, 2), dtype='float32')     # TRAFFIC_CONVENTION_LEN = 2
            Xin3   = np.zeros((yuvX_len-1, 512), dtype='float32')   # rnn state
            
            Y_shapes = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]
            Y = [np.zeros((yuvX_len-1, s), dtype='float32') for s in Y_shapes]
              
            rnn_st_next = Xin3[0]  

            for i in tqdm(range(yuvX_len-1)):
              
                vsX1 = yuvX[i]
                vsX2 = yuvX[i+1]

                Ximgs[i] = np.vstack((vsX1, vsX2))  
        
                Xin3[i] = rnn_st_next


                outs = supercombo_y(Ximgs[i], Xin1[i], Xin2[i], Xin3[i])
                
              
                rnn_st_next = outs[11][0]
                  
                assert len(outs) == 12

                for j in range(len(outs)):
                    Y[j][i] = outs[j]

            
            data = {'Ximgs': Ximgs, 'Xin1': Xin1, 'Xin2': Xin2, 'Xin3': Xin3, 'Y': Y}

            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)    
                print(f'pkl_file: {pkl_file}')          



            with open(pkl_file, 'rb') as f:            
                data_rd = pickle.load(f)        
                # assert (data_rd['Ximgs'] == Ximgs).all()
                assert data_rd['Ximgs'].shape == (yuvX_len-1, 12, 128, 256)
                assert data_rd['Xin1'].shape == (yuvX_len-1, 8)
                assert data_rd['Xin2'].shape == (yuvX_len-1, 2)
                assert data_rd['Xin3'].shape == (yuvX_len-1, 512)
                print(f'X all verified.')

                for i in range(12):
                    assert data_rd['Y'][i].shape == (yuvX_len-1, Y_shapes[i])
                    # assert (data_rd['Y'][i] == Y[i]).all()
                    print(f'Y[{i}] verified.')


if __name__ == "__main__":
  #camera_file = '/home/richard/dataB6/UHD--2018-08-02--08-34-47--33/video.hevc'
  #Xins, Ytrue = datagen_test(1, camera_file)  # for Testing in train_modelB6.py
  # camera_file = '/home/richard/dataB6/UHD--2018-08-02--08-34-47--33/yuv.h5'
  # datagen_debug(100, camera_file)  # for parallel GPUs (batches) in def datagen
    # batch_size=100 > 1200/100 > Steps = 11

  datagen()
  # for i in a:
  #   1+1
