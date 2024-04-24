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

from tools.lib.framereader import FrameReader
from common.transformations.model import medmodel_intrinsics
from cameraB3 import transform_img, eon_intrinsics
from PIL import Image
import glob

#physical_gpus = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_gpus[0], True)

os.environ['CUDA_VISIBLE_DEVICES'] = ""

def supercombo_y(Ximgs, Xin1, Xin2, Xin3):
  supercombo = load_model('saved_model/supercombo079.keras', compile=False)
    #--- np.shape(Ximgs) = (12, 128, 256)
  Ximgs = np.expand_dims(Ximgs, axis=0)
    #--- np.shape(Ximgs) = (1, 12, 128, 256)
    # x = np.array([1, 2]) > x.shape: (2,) > y = np.expand_dims(x, axis=0) > y: array([[1, 2]]) > y.shape: (1, 2)
  Xin1 = np.expand_dims(Xin1, axis=0)
  Xin2 = np.expand_dims(Xin2, axis=0)
  Xin3 = np.expand_dims(Xin3, axis=0)
  inputs = [Ximgs, Xin1, Xin2, Xin3]

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




def datagen(batch_size, camera_files):
  print('################################')
  # 2 yub imgs.
  Ximgs  = np.zeros((batch_size, 12, 128, 256), dtype='float32')   # Is YUV input img uint8? No. See float8 ysf = convert_float8(ys) in loadyuv.cl

  Ximgs_rgb  = np.zeros((batch_size, 256, 256, 3), dtype='float32') 


  Xin1   = np.zeros((batch_size, 8), dtype='float32')     # DESIRE_LEN = 8
  Xin2   = np.zeros((batch_size, 2), dtype='float32')     # TRAFFIC_CONVENTION_LEN = 2
  Xin3   = np.zeros((batch_size, 512), dtype='float32')   # rnn state
  
  Ytrue0 = np.zeros((batch_size, 385), dtype='float32')
  Ytrue1 = np.zeros((batch_size, 386), dtype='float32')
  Ytrue2 = np.zeros((batch_size, 386), dtype='float32')
  Ytrue3 = np.zeros((batch_size, 58), dtype='float32')
  Ytrue4 = np.zeros((batch_size, 200), dtype='float32')
  Ytrue5 = np.zeros((batch_size, 200), dtype='float32')
  Ytrue6 = np.zeros((batch_size, 200), dtype='float32')
  Ytrue7 = np.zeros((batch_size, 8), dtype='float32')
  Ytrue8 = np.zeros((batch_size, 4), dtype='float32')
  Ytrue9 = np.zeros((batch_size, 32), dtype='float32')
  Ytrue10 = np.zeros((batch_size, 12), dtype='float32')
  Ytrue11 = np.zeros((batch_size, 512), dtype='float32')

  Xin1[:, 0] = 1.0   # go straight? desire_state_prob[0] = 1.0
  Xin2[:, 0] = 1.0   # traffic_convention[0] = 1.0 = left hand drive like in Taiwan


  for cfile in camera_files: assert os.path.isfile(cfile)

  hevc_files = glob.glob("/home/richard/dataB6/*/video.hevc")
  yuv_files = glob.glob("/home/richard/dataB6/*/yuv.h5")
  
  

  print(f'hevc_files: {hevc_files}')
  print(f'yuv_files: {yuv_files}')



  Epoch = 1
  Xin3_temp = Xin3[0]  #--- np.shape(Xin3_temp) = (512,)

  

  # for cfile in camera_files:

  #     # yuvh5f = h5py.File(cfile, 'r')
  #     # print(f"yuvh5f['X'].shape: {yuvh5f['X'].shape}")
  #     # exit()

  #     with h5py.File(cfile, "r") as yuv:            
  #         yuvX = np.copy(yuv['X']) #--- yuvX.shape = (1200, 6, 128, 256) or (1199, 6, 128, 256)        
 
  while True:

      CFile = 1
      Step = 1

      # for cfile in camera_files:
      for i in range(len(camera_files)):

          yuv_file = yuv_files[i]
          hevc_file = hevc_files[i]


          with h5py.File(yuv_file, "r") as yuv:     

              yuvX = yuv['X'] #--- yuvX.shape = (1200, 6, 128, 256) or (1199, 6, 128, 256)        
       



              frames_rgb = []

              cap = cv2.VideoCapture(hevc_file)

              ret, frame = cap.read()  # frame; rgb img.
              count = 0
              while ret: 
                  count += 1   

                  frame = Image.fromarray(frame)
                  frame = frame.resize((256, 256), Image.BILINEAR) 
                  frame = np.array(frame) # (84, 84, 3)

                  frames_rgb.append(frame)

                  ret, frame = cap.read() 

        




                  

              yuvX_len = len(yuvX)  # = 1200 or 1199
              lastIdx = yuvX_len - 2 - batch_size   # cannot be the last frame yuvX_len-1

              for i in range(0, lastIdx, batch_size):

                  print("#--- Epoch:", Epoch, " Step:", Step, " CFile:", CFile)

                  bcount = 0

                  while bcount < batch_size:
                      
                      vsX1 = yuvX[bcount+i]
                      vsX2 = yuvX[bcount+i+1]

                      # rgb = frames_rgb[bcount+i]

                        #--- vsX2.shape = (6, 128, 256)
                      Ximgs[bcount] = np.vstack((vsX1, vsX2))   # stack two yuv images i and i+1
                        #--- Ximgs[bcount].shape = (12, 128, 256)
                      
                      Ximgs_rgb[bcount] = frames_rgb[bcount+i]

                      Xin3[bcount] = Xin3_temp

                      outs = supercombo_y(Ximgs[bcount], Xin1[bcount], Xin2[bcount], Xin3[bcount])
                        #--- len(outs) = 12
                      Xin3_temp = outs[11][0]
                        #--- np.shape(outs[11][0]) = (512,)  np.shape(outs[11]) = (1, 512)

                      Ytrue0[bcount] = outs[0]
                      Ytrue1[bcount] = outs[1]
                      Ytrue2[bcount] = outs[2]
                      Ytrue3[bcount] = outs[3]
                      Ytrue4[bcount] = outs[4]
                      Ytrue5[bcount] = outs[5]
                      Ytrue6[bcount] = outs[6]
                      Ytrue7[bcount] = outs[7]
                      Ytrue8[bcount] = outs[8]
                      Ytrue9[bcount] = outs[9]
                      Ytrue10[bcount] = outs[10]
                      Ytrue11[bcount] = outs[11]
                      bcount += 1

                  yield Ximgs_rgb, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, \
                            Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11
                  
                  Step += 1

          CFile += 1

      Epoch += 1




















def datagen_test(batch_size, cfile):  # Test model by only 2 images
  Ximgs  = np.zeros((batch_size, 12, 128, 256), dtype='float32')
  Xin1   = np.zeros((batch_size, 8), dtype='float32')
  Xin2   = np.zeros((batch_size, 2), dtype='float32')
  Xin3   = np.zeros((batch_size, 512), dtype='float32')
  Ytrue0 = np.zeros((batch_size, 385), dtype='float32')
  Ytrue1 = np.zeros((batch_size, 386), dtype='float32')
  Ytrue2 = np.zeros((batch_size, 386), dtype='float32')
  Ytrue3 = np.zeros((batch_size, 58), dtype='float32')
  Ytrue4 = np.zeros((batch_size, 200), dtype='float32')
  Ytrue5 = np.zeros((batch_size, 200), dtype='float32')
  Ytrue6 = np.zeros((batch_size, 200), dtype='float32')
  Ytrue7 = np.zeros((batch_size, 8), dtype='float32')
  Ytrue8 = np.zeros((batch_size, 4), dtype='float32')
  Ytrue9 = np.zeros((batch_size, 32), dtype='float32')
  Ytrue10 = np.zeros((batch_size, 12), dtype='float32')
  Ytrue11 = np.zeros((batch_size, 512), dtype='float32')

  Xin1[:, 0] = 1.0
  Xin2[:, 0] = 1.0

  if os.path.isfile(cfile):
    print('#--- datagen_test  OK: cfile exists')
  else:
    print('#--- datagen_test  Error: cfile does not exist')

  fr = FrameReader(cfile)
  frame_count = fr.frame_count
  print('#--- frame_count =', frame_count)
    #--- frame_count = 1199
  cap = cv2.VideoCapture(cfile)
    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #--- total_frames = -192153584101141.0

  CsYUVs = []
  for i in range(2):
    ret, RGB = cap.read()
    if not ret:
      print("Error: Can't receive frame (stream end?). Exiting ...")
      break
    bYUV = cv2.cvtColor(RGB, cv2.COLOR_BGR2YUV_I420)  # BGR is slightly different from RGB
    sYUV = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                         output_size=(512, 256))   # resize bYUVs to small YUVs
    CsYUV = sYUV_to_CsYUV(sYUV)
      #---            RGB.shape,    bYUV.shape,  sYUV.shape, np.shape(CsYUV) =
      # (H, W, C) = (874, 1164, 3) (1311, 1164)  (384, 512)   (6, 128, 256) = (C, H, W)
    CsYUVs.append(CsYUV)

    ''' Show video images
    window_name = 'RGB Frame # ' + str(i)
    cv2.imshow(window_name, RGB)
    cv2.waitKey(1000)
      # Convert YUV420 to Grayscale
    gray = cv2.cvtColor(sYUV, cv2.COLOR_YUV2GRAY_I420)
    cv2.imshow("yuv2gray", gray)
    cv2.waitKey(1000)
      # Convert bYUV back to RGB
    rgbB = cv2.cvtColor(bYUV, cv2.COLOR_YUV2RGB_I420)
    cv2.imshow("yuvB2rgb", rgbB)
    cv2.waitKey(1000)
      # Convert sYUV back to RGB
    rgbS = cv2.cvtColor(sYUV, cv2.COLOR_YUV2RGB_I420)
    cv2.imshow("yuvS2rgb (256x512)", rgbS)
    cv2.waitKey(1000)

    input("Press ENTER to close RGB Frame # ...")
    if cv2.waitKey(1000) == 27:   # if ENTER is pressed
      cv2.destroyAllWindows()
    cv2.destroyAllWindows() '''
  cap.release()

  Xin3_temp = Xin3[0]
  bcount = 0
  while bcount < batch_size:
    Ximgs[bcount] = np.vstack((CsYUVs[0], CsYUVs[1]))
    outs = supercombo_y(Ximgs[bcount], Xin1[bcount], Xin2[bcount], Xin3[bcount])
    Xin3_temp = outs[11][0]
    Ytrue0[bcount] = outs[0]
    Ytrue1[bcount] = outs[1]
    Ytrue2[bcount] = outs[2]
    Ytrue3[bcount] = outs[3]
    Ytrue4[bcount] = outs[4]
    Ytrue5[bcount] = outs[5]
    Ytrue6[bcount] = outs[6]
    Ytrue7[bcount] = outs[7]
    Ytrue8[bcount] = outs[8]
    Ytrue9[bcount] = outs[9]
    Ytrue10[bcount] = outs[10]
    Ytrue11[bcount] = outs[11]
    bcount += 1
  Xins  = [Ximgs, Xin1, Xin2, Xin3]
  Ytrue = np.hstack((Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11))
  return Xins, Ytrue



def datagen_debug(batch_size, cfile):
  Step = 1
  with h5py.File(cfile, "r") as yuv:
    yuvX = yuv['X']
    print('#--- yuvX.shape =', yuvX.shape)
    yuvX_len = len(yuvX)
    lastIdx = yuvX_len - 2 - batch_size
    print('#--- lastIdx =', lastIdx)
    for i in range(0, lastIdx, batch_size):
      print("#--- Epoch:", 1, " Step:", Step, " CFile:", 1, " i:", i)
      bcount = 0
      while bcount < batch_size:
        bcount += 1
      Step += 1
  return


if __name__ == "__main__":
  #camera_file = '/home/richard/dataB6/UHD--2018-08-02--08-34-47--33/video.hevc'
  #Xins, Ytrue = datagen_test(1, camera_file)  # for Testing in train_modelB6.py
  # camera_file = '/home/richard/dataB6/UHD--2018-08-02--08-34-47--33/yuv.h5'
  # datagen_debug(100, camera_file)  # for parallel GPUs (batches) in def datagen
    # batch_size=100 > 1200/100 > Steps = 11

  camera_files = glob.glob("/home/richard/dataB6/*/yuv.h5")

  print(f'################# camera_files{ camera_files}')
  
  datagen(1, camera_files)