'''  JLL, 2021.9.24 - 2022.5.17, 2024.2.6
for radarState.leadOne; radarState.leadTwo in bz2toh5.py; 240124
Convert video.hevc to yuv.h5
  hevc =>  bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=>  CbYUV (6, 291, 582) = (C, H, W) [key: 1311 =  874x3/2]
           sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=>  CsYUV (6, 128, 256) = (C, H, W) [key:  384 =  256x3/2]

(sconsvenv) richard@Liu:~/openpilot/aJLL/Model$ python hevc2yuvh5.py
Input:
  /home/richard/dataB6/UHD--2018-08-02--08-34-47--32/video.hevc
  /home/richard/dataB6/UHD--2018-08-02--08-34-47--33/video.hevc
  /home/richard/dataB6/UHD--2018-08-02--08-34-47--37/video.hevc
Output:
  /home/richard/dataB6/UHD--2018-08-02--08-34-47--32/yuv.h5  (1200 frames)
  /home/richard/dataB6/UHD--2018-08-02--08-34-47--33/yuv.h5
  /home/richard/dataB6/UHD--2018-08-02--08-34-47--37/yuv.h5
'''
import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from tools.lib.framereader import FrameReader
from common.transformations.model import medmodel_intrinsics
from cameraB3 import transform_img, eon_intrinsics
  # cameraB3 = /home/richard/YPN/Leon/common/transformations/camera.py

import glob
from PIL import Image 



def RGB_to_sYUVs(video, frame_count):
  bYUVs = []
  for i in range(frame_count):


    # frame; rgb img.
    ret, frame = video.read()  #--- ret =  True
      #--- frame.shape = (874, 1164, 3) = (H, W, C) = (row, col, dep) = (axis0, axis1, axis2)

    # print(f'frame.shape: {frame.shape}')

    # save_frame('output', 'test.jpeg', frame)

    # exit()

    #--- np.shape(bYUV) = (1311, 1164)
    bYUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)  # BGR is slightly different from RGB

      
    bYUVs.append(bYUV.reshape((874*3//2, 1164))) # no effect, has same shape before and after.
    
      #--- bYUV.shape = (1311, 1164) # TFNs = 874*1164*3/2 bytes
      #--- np.shape(bYUVs) = (10, 1311, 1164)


  sYUVs = np.zeros((len(bYUVs), 384, 512), dtype=np.uint8) # np.uint8 = 0~255

  for i, img in tqdm(enumerate(bYUVs)):
    sYUVs[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
                             yuv=True, output_size=(512, 256))  # (W, H)
      #--- np.shape(sYUVs) = (10, 384, 512)
  return sYUVs



def sYUVs_to_CsYUVs(sYUVs):
  H = (sYUVs.shape[1]*2)//3  # 384x2//3 = 256
  W = sYUVs.shape[2]         # 512
  CsYUVs = np.zeros((sYUVs.shape[0], 6, H//2, W//2), dtype=np.uint8)

  CsYUVs[:, 0] = sYUVs[:, 0:H:2, 0::2]  # [2::2] get every even starting at 2
  CsYUVs[:, 1] = sYUVs[:, 1:H:2, 0::2]  # [start:end:step], [2:4:2] get every even starting at 2 and ending at 4
  CsYUVs[:, 2] = sYUVs[:, 0:H:2, 1::2]  # [1::2] get every odd index, [::2] get every even
  CsYUVs[:, 3] = sYUVs[:, 1:H:2, 1::2]  # [::n] get every n-th item in the entire sequence
  CsYUVs[:, 4] = sYUVs[:, H:H+H//4].reshape((-1, H//2, W//2))
  CsYUVs[:, 5] = sYUVs[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))

  CsYUVs = np.array(CsYUVs).astype(np.float32)
    #--- np.shape(CsYUVs) = (10, 6, 128, 256)
  return CsYUVs






 
def save_frame(dir_name, name, frame):
    if(not os.path.exists(dir_name)):
        os.mkdir(dir_name)    
    
    # print(f'############# frame.shape: {frame.shape}')
    frame = np.squeeze(frame)
    img = Image.fromarray(frame)
    img.save(os.path.join(dir_name, name))        








def getFrame_rgb():
    
    # print(f'glob: {glob.glob("/home/richard/dataB6/*/video.hevc")}')

    all_videos = glob.glob("/home/richard/dataB6/*/video.hevc")

    for vi in all_videos:

        # fr = FrameReader(vi)
        # frame_count = fr.frame_count # 1199
        
        cap = cv2.VideoCapture(vi)

        # sYUVs = RGB_to_sYUVs(cap, frame_count)
  
        
        ret, frame = cap.read()  #--- ret =  True   
        count = 0
        while ret: 
            count += 1   

            img = Image.fromarray(frame)
            img = img.resize((256, 256), Image.BILINEAR) 
            img = np.array(img) # (84, 84, 3)

            # save_frame('output', 'test.jpeg', img)
            print(f'img.shape: {img.shape}')
            ret, frame = cap.read() 
            
            # print(f'count: {count}')
            exit()



def makeYUV(all_dirs):
  #all_videos = ['/home/richard/dataB/'+i+'/video.hevc' for i in all_dirs]
  all_videos = ['/home/richard/dataB6/'+i+'/video.hevc' for i in all_dirs]

  
  for vi in all_videos:
    # yuvh5 = vi.replace('video.hevc','yuv.h5')
    yuvh5 = vi.replace('video.hevc','test.h5')
    print("#--- video =", vi)
 

    if not os.path.isfile(yuvh5):
      fr = FrameReader(vi)
      frame_count = fr.frame_count
      print('#--- frame_count =', frame_count)
      cap = cv2.VideoCapture(vi)

      with h5py.File(yuvh5, 'w') as h5f:
        h5f.create_dataset('X', (frame_count, 6, 128, 256))
          #h5f.create_dataset('X', (1150, 6, 128, 256))

        sYUVs = RGB_to_sYUVs(cap, frame_count)

        CsYUVs = sYUVs_to_CsYUVs(sYUVs)

        for i in range(frame_count):
          h5f['X'][i] = CsYUVs[i]

        print("#--- yuv.h5 created ...")
        yuvh5f = h5py.File(yuvh5, 'r') # read .h5, 'w': write
            #--- yuvh5f.keys() = <KeysViewHDF5 ['X']>
            #--- yuvh5f['X'].shape = (1199, 6, 128, 256)
        # print(f"yuvh5f['X'].shape: {yuvh5f['X'].shape}")
        # exit()



    else:
      print("#--- yuv.h5 exists ...")
      yuvh5f = h5py.File(yuvh5, 'r')  # read .h5, 'w': write



if __name__ == "__main__":
  #all_dirs = os.listdir('/home/richard/dataB')
  all_dirs = os.listdir('/home/richard/dataB6')
  # makeYUV(all_dirs)
  getFrame_rgb()
