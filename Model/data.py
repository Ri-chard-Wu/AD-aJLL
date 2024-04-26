 
import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm 
from common.transformations.model import medmodel_intrinsics
from cameraB3 import transform_img, eon_intrinsics

import glob
from PIL import Image 

 
import random 
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
 
import pickle
import time

   
from lanes_image_space import transform_points, transform_point

from parserB6 import parser


from multiprocessing import Process, Pipe




  





 
def plot_path(outs, frame, dir_name, file_name):

    if(not os.path.exists(dir_name)):
        os.mkdir(dir_name)
        
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

    # print(f"parsed['lead_xyva'][0].shape: {parsed['lead_xyva'][0].shape}")
    x, y, v, a = parsed['lead_xyva'][0]
    x, y = transform_point(x, y)


    # print(f'new_x_path, {new_x_path}')
    # print(f'new_y_path, {new_y_path}')

    # print(f'x, y: {x}, {y}')

    plt.plot([x], [y], label='transformed', marker='^', color='y', markersize=10)
 
    plt.plot(new_x_left, new_y_left, label='transformed', color='r')
    plt.plot(new_x_path, new_y_path, label='transformed', color='g')
    plt.plot(new_x_right, new_y_right, label='transformed', color='b')

    plt.imshow(frame)   # Merge raw image and plot together

    plt.subplot(222)
    plt.gca().invert_yaxis()
    plt.title("Camera View")
 
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

    
    plt.savefig(dir_name + '/' + file_name)  







def read_hevc(hevc_file):

    frames_rgb = []

    cap = cv2.VideoCapture(hevc_file)

    ret, frame = cap.read()  # bgr img.
     
    count = 0
    while ret: 
        count += 1   
 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # rgb img.
        frames_rgb.append(frame)

        ret, frame = cap.read() 
  
    return frames_rgb







 
os.environ['CUDA_VISIBLE_DEVICES'] = ""


# supercombo = load_model('saved_model/supercombo079.keras', compile=False)  
# supercombos = [load_model('saved_model/supercombo079.keras', compile=False)  for i in range(8)]

def worker(tid, remote, parent_remote, videos):
 
    print(f'[{tid}] loading model...')
    
    # supercombo = supercombos[tid]
    supercombo = load_model('saved_model/supercombo079.keras', compile=False)

    print(f'[{tid}] done loading model...')

    parent_remote.close()
   

    for idx, vi in enumerate(videos):

        output_file = vi.replace('fcamera.hevc', 'data.pkl') #.replace('/TData1/','/TData1-pp/') 
            
        cap = cv2.VideoCapture(vi)    

        ret, frame = cap.read() 
        YUV_next = RGB_to_YUV(frame)
        ret, frame = cap.read() 

        T = 1200

        Xin3 = np.zeros((T, 512))
        rnn_st_next = Xin3[0]

        Y_shapes = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]
        Y = [np.zeros((T, s), dtype='float32') for s in Y_shapes]
    
        t = 0
        while ret:
            if(t % 50 == 0):
                print(f'[{tid}] t: {t}')

            YUV = YUV_next 
            YUV_next = RGB_to_YUV(frame)

            Xin3[t] = rnn_st_next
 
            Ximg = np.vstack((YUV, YUV_next))[None]
            assert Ximg.shape == (1, 12, 128, 256)
  
       
            outs = supercombo([Ximg, np.zeros((1, 8)), np.zeros((1, 2)), rnn_st_next[None]])

            if(t % 10 == 0):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plot_path(outs, frame, dir_name=f'output/gen/{tid}', file_name=f'gen-{t}.png')
   

    


            rnn_st_next = outs[11][0] # (512, )

            assert len(outs) == 12

            for i in range(len(outs)):
                Y[i][t] = outs[i]


            t+=1
            ret, frame = cap.read() 

        cap.release()


        data = {'Xin3': Xin3, 'Y': Y}

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)    
                     
 
        with open(output_file, 'rb') as f:            
            data_rd = pickle.load(f)        
            assert (data_rd['Xin3'] == Xin3).all() 
            
            for i in range(12):
                assert (data_rd['Y'][i] == Y[i]).all()
            

        print(f'[{tid}] completed {idx+1} / {len(videos)}: {output_file}') 



class VecEnv():
 
    def __init__(self, n_workers, video_batchs):
        
        self.closed = False 
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_workers)])

        self.ps = [Process(target=worker, args=(i, work_remote, remote, video_batch))
                   for i, (work_remote, remote, video_batch) in\
                    enumerate(zip(self.work_remotes, self.remotes, video_batchs))]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
 

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True









def supercombo_y(Ximgs, Xin1, Xin2, Xin3):
      
  Ximgs = np.expand_dims(Ximgs, axis=0) # (1, 12, 128, 256)        
  Xin1 = np.expand_dims(Xin1, axis=0)
  Xin2 = np.expand_dims(Xin2, axis=0)
  Xin3 = np.expand_dims(Xin3, axis=0)
  inputs = [Ximgs, Xin1, Xin2, Xin3]

  outputs = supercombo(inputs)

  return outputs



 



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


def save_frame(dir_name, name, frame):
    if(not os.path.exists(dir_name)):
        os.mkdir(dir_name)    
    
    frame = np.squeeze(frame)
    img = Image.fromarray(frame)
    img.save(os.path.join(dir_name, name))        


def datagen(rnn_st, YUVs, YUVs_next):    

    Y_shapes = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]
    Y = [np.zeros((s,), dtype='float32') for s in Y_shapes]
  
    # Xin3 = np.zeros((512,))


    # # rnn_st_next = Xin3[0]  
    # rnn_st_next = Xin3[0] 

    for i in tqdm(range(yuvX_len-1)):
        
        vsX1 = yuvX[i]
        vsX2 = yuvX[i+1]

        # X[0][i] = frames_rgb[i]

        # Ximgs[i] = np.vstack((vsX1, vsX2))  
        Ximg = np.vstack((vsX1, vsX2))  

        

        outs = supercombo_y(Ximg, np.zeros((8, )), np.zeros((2, )), rnn_st)
          

        rnn_st_next = outs[11][0]
            
        assert len(outs) == 12

        for j in range(len(outs)):
            Y[i] = outs[j]


    # data = {'Ximgs': Ximgs, 'Xin1': Xin1, 'Xin2': Xin2, 'Xin3': Xin3, 'Y': Y}
    data = {'Xin3': Xin3, 'Y': Y}

    # with open(output_file, 'wb') as f:
    #     pickle.dump(data, f)    
    #     print(f'generated output_file: {output_file}')          

 
    # with open(output_file, 'rb') as f:            
    #     data_rd = pickle.load(f)        

    #     assert (data_rd['Xin3'] == Xin3).all()


    #     # assert (data_rd['Ximgs'] == Ximgs).all()
    #     # assert data_rd['Ximgs'].shape == (yuvX_len-1, 12, 128, 256)
    #     # assert data_rd['Xin1'].shape == (yuvX_len-1, 8)
    #     # assert data_rd['Xin2'].shape == (yuvX_len-1, 2)
    #     # assert data_rd['Xin3'].shape == (yuvX_len-1, 512)

    #     print(f'Xin3 verified.')

    #     for i in range(12):
    #         assert data_rd['Y'][i].shape == (yuvX_len-1, Y_shapes[i])
    #         assert (data_rd['Y'][i] == Y[i]).all()
    #         print(f'Y[{i}] verified.')






if __name__ == "__main__":
    #all_dirs = os.listdir('/home/richard/dataB')
    all_dirs = os.listdir('/home/richard/dataB6')
    # makeYUV(all_dirs)



    all_videos = glob.glob("/home/richard/Downloads/TData1/*.hevc")
    # print(f'len(all_videos): {len(all_videos)}')

    # chunk_size = len(all_videos) // 8

    # makeYUV(all_videos[3:10])
    # getFrame_rgb()

    n_workers = 8
    video_batchs = [[] for i in range(n_workers)]

    for i in range(len(all_videos)):
        video_batchs[i % n_workers].append(all_videos[i])
    
    
    print(f'load dists: {[len(video_batch) for video_batch in video_batchs]}')
    
    VecEnv(n_workers, video_batchs)
    
    while(1):
        a = 1
        


    # batch_size = 8
    # n_batchs = len(all_videos) // batch_size
    
    # # for vi in all_videos:
    # for i in range(n_batchs):        

    #     batch = all_videos[i*batch_size:(i+1)*batch_size]

    #     # yuvh5 = vi.replace('video.hevc','yuv.h5')
    #     # frames_rgb = read_hevc(vi)

    #     # yuvh5 = vi.replace('fcamera.hevc','yuv.h5')
    #     # yuvh5 = yuvh5.replace('/TData1/','/TData1-pp/')

    #     output_files = [vi.replace('fcamera.hevc', 'data.pkl').replace('/TData1/','/TData1-pp/') for vi in batch]
        
    #     # yuvh5 = vi.replace('video.hevc','test.h5')
    #     # print("## video =", vi)
    
    #     caps = [cv2.VideoCapture(vi) for vi in batch]    

    #     rets, frames = zip(*[cap.read() for cap in caps])
    #     YUVs_next = [RGB_to_YUV(frame) for frame in frames] 

    #     rets, frames = zip(*[cap.read() for cap in caps])
    #     # if not np.array(rets).all(): continue
    #     # YUVs_next = [RGB_to_YUV(frame) for frame in frames] 

    #     rnn_sts = np.zeros((batch_size, 512))

    #     Xin3s = [[] for i in range(batch_size)]
    #     Ys = [[] for i in range(batch_size)]

    #     while np.array(rets).any():

    #         # sYUVs = [RGB_to_sYUVs(frame) for frame in frames] 
    #         # CsYUVs = [sYUVs_to_CsYUVs(sYUV) for sYUV in sYUVs]
    #         YUVs = YUVs_next
    #         # for i in range(batch_size):
    #         #     frame = frames[i]
    #         #     if(not rets[i]): frame = YUVs[i]
    #         #     YUVs_next = [RGB_to_YUV(frame) for frame in frames] 

    #         YUVs_next = [frames[i] if rets[i] else YUVs[i] for i in range(batch_size)]

    #         # vsX1 = yuvX[i]
    #         # vsX2 = yuvX[i+1]

    #         # # X[0][i] = frames_rgb[i]

    #         # # Ximgs[i] = np.vstack((vsX1, vsX2))  
    #         # Ximg = np.vstack((vsX1, vsX2)) 
 
    #         Ximgs = np.concatenate([np.vstack((YUVs[i], YUVs_next[i]))[None] for i in range(batch_size)])
    #         assert Ximgs.shape == (b, 12, 128, 256)
  
    #         outs = supercombo([Ximgs, np.zeros((batch_size, 8)), np.zeros((batch_size, 2)), rnn_sts])
  
            
    #         for i in range(batch_size):
    #             Xin3s[i].append(rnn_sts[i])
    #             Ys[i].append(outs[i])





    #         rnn_sts = outs[11][:] # (batch_size, 512)
                
    #         assert len(outs) == 12

    #         for j in range(len(outs)):
    #             Y[i] = outs[j]


    #         # outs = supercombo_y(Ximg, np.zeros((8, )), np.zeros((2, )), rnn_st)

    #         rets, frames = zip(*[cap.read() for cap in caps])


    #     for cap in caps: cap.release()

    # # models = [load_model('saved_model/supercombo079.keras', compile=False) for i in range(8)]

    # # while(1):
    # #     for i in range(8):
    # #         models[i]([[np.zeros((1, 12, 128, 256)), np.zeros((1, 8)), np.zeros((1, 2)), np.zeros((1, 512))]])
    # #     print('passed')



