 
import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm 
from common.transformations.model import medmodel_intrinsics


import glob
from PIL import Image 

 
import random 
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
 
import pickle
import time

   
from cameraB3 import transform_img, eon_intrinsics, draw_path   
from lanes_image_space import transform_points, transform_point

from parserB6 import parser


from multiprocessing import Process, Pipe




  
 
os.environ['CUDA_VISIBLE_DEVICES'] = ""



Y_shape = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]


 
def plot_outs(outs, frame, dir_name, file_name):

    try:

        frame = frame.astype(np.uint8)

        # print(f'np.mean(frame): {np.mean(frame)}')
        # exit()
        if(not os.path.exists(dir_name)):
            # os.mkdir(dir_name)
            os.makedirs(dir_name, exist_ok=True)

        PATH_DISTANCE = 192
        x_lspace = np.linspace(1, PATH_DISTANCE, PATH_DISTANCE)  
        
        # outs = [a.numpy() for a in outs]
    
        parsed = parser(outs)
        
        #--- len(parsed) = 25
        #[print("#--- parsed[", x, "].shape =", parsed[x].shape) for x in parsed]   # see output.txt
        
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # cv2 reads images in BGR format (instead of RGB)

        plt.clf()   # clear figure
        plt.xlim(0, 1200)
        plt.ylim(800, 0)



        # -----------------------

        plt.subplot(221)   # 221: 2 rows, 2 columns, 1st sub-figure

        
        # print(f"parsed['path_valid_len']: {parsed['path_valid_len']}")

        l = int(parsed['path_valid_len'])   
        
        # print(f"parsed['path'][0][:l]: {parsed['path'][0][:l]}") 

        plt.imshow(draw_path(frame.copy(), parsed["path"][0][:l], x_lspace[:l]))

        # plt.imshow(frame)

        # new_x_path, new_y_path = transform_points(x_lspace[:l], parsed["path"][0][:l]) 
        # plt.plot(new_x_path, new_y_path, color='g')



        plt.title(f"Overlay (truncated) l: {l}")


        L_ll = int(parsed['lll_valid_len']) 
        new_x_left, new_y_left = transform_points(x_lspace, parsed["lll"][0]) 

        L_rl = int(parsed['rll_valid_len']) 
        new_x_right, new_y_right = transform_points(x_lspace, parsed["rll"][0])

        plt.plot(new_x_left[:L_ll], new_y_left[:L_ll], label='transformed', color='r')    
        plt.plot(new_x_right[:L_rl], new_y_right[:L_rl], label='transformed', color='r')

        # print(f"parsed['lead_xyva'][0].shape: {parsed['lead_xyva'][0].shape}")
        x, y, v, a = parsed['lead_xyva'][0]
        x, y = transform_point(x, y) 
        plt.plot([x], [y],  marker='^', color='y', markersize=10)
    
        # -----------------------

        plt.subplot(222)   # 221: 2 rows, 2 columns, 1st sub-figure
        plt.title(f"Overlay (no truncate)")
        
        plt.imshow(draw_path(frame.copy(), parsed["path"][0], x_lspace))
        plt.plot(new_x_left, new_y_left, color='r')    
        plt.plot(new_x_right, new_y_right, color='r')
    

        # -----------------------

        plt.subplot(223)
        plt.title("Original Scene")
        plt.imshow(frame)
        # -----------------------

        plt.subplot(224)
        plt.gca().invert_xaxis()

        # Needed to invert axis because standard left lane is positive and right lane is negative, so we flip the x axis
        plt.title("Top-Down View (truncated)")
        plt.plot(parsed["lll"][0][:L_ll], range(0, L_ll), "r-", linewidth=1)
        plt.plot(parsed["path"][0][:l], range(0, l), "g-", linewidth=1)
        plt.plot(parsed["rll"][0][:L_rl], range(0, L_rl), "b-", linewidth=1)

        plt.tight_layout()
        plt.savefig(dir_name + '/' + file_name)  
        
    except:
        pass


 

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






# supercombo = load_model('saved_model/supercombo079.keras', compile=False)  
# supercombos = [load_model('saved_model/supercombo079.keras', compile=False)  for i in range(8)]

def worker(tid, remote, parent_remote, videos):
 
    print(f'[{tid}] loading model...')
    
    # supercombo = supercombos[tid]
    supercombo = load_model('saved_model/supercombo079.keras', compile=False)

    print(f'[{tid}] done loading model...')

    if parent_remote is not None:
        parent_remote.close()
   

    for idx, vi in enumerate(videos):

        output_file = vi.replace('fcamera.hevc', 'data.pkl') #.replace('/TData1/','/TData1-pp/') 
        if os.path.isfile(output_file): 
            print(f'[{tid}] file exists: {output_file}. skip.')
            continue


        cap = cv2.VideoCapture(vi)    

        ret, frame = cap.read() 
        YUV_next = RGB_to_YUV(frame)
        ret, frame = cap.read() 

        # T = 1200

        # Xin3 = np.zeros((T, 512))
        # rnn_st_next = Xin3[0]
        
        Xin3 = []
        rnn_st_next = np.zeros((512, ))
        

        # Y_shape = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]
        # Y = [np.zeros((T, s), dtype='float32') for s in Y_shapes]
        Y = [[] for i in range(12)]
    
        t = 0
        while ret:
            # if(t % 50 == 0):
            #     print(f'[{tid}] t: {t}')
            # if(t>10):break
            YUV = YUV_next 
            YUV_next = RGB_to_YUV(frame)

            # Xin3[t] = rnn_st_next
            Xin3.append(rnn_st_next[None])

 
            Ximg = np.vstack((YUV, YUV_next))[None]
            assert Ximg.shape == (1, 12, 128, 256)
  
       
            outs = supercombo([Ximg, np.zeros((1, 8)), np.zeros((1, 2)), rnn_st_next[None]])

            if(t % 10 == 0):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                
                plot_outs([a.numpy() for a in outs], frame, dir_name=f'output/gen/{tid}', file_name=f'gen-{t}.png')
   

    


            rnn_st_next = outs[11][0] # (512, )

            assert len(outs) == 12

            for i in range(len(outs)):
                # Y[i][t] = outs[i]
                Y[i].append(outs[i])


            t+=1
            ret, frame = cap.read() 

        cap.release()

        Xin3 = np.concatenate(Xin3)
    # n_workers = 8
    # video_batchs = [[] for i in range(n_workers)]

    # for i in range(len(all_videos)):
    #     video_batchs[i % n_workers].append(all_videos[i])
    
    
    # print(f'load dists: {[len(video_batch) for video_batch in video_batchs]}')
    
    # VecEnv(n_workers-1, video_batchs[:-1])
    
    
    # worker(n_workers-1, None, None, video_batchs[n_workers-1])
        
 
        assert Xin3.shape == (t, 512)

        Y = [np.concatenate(y) for y in Y]

        data = {'Xin3': Xin3, 'Y': Y}

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)    
                     
 
        with open(output_file, 'rb') as f:            
            data_rd = pickle.load(f)        
            assert (data_rd['Xin3'] == Xin3).all() 
            
            for i in range(12):
                assert (data_rd['Y'][i] == Y[i]).all()
                assert data_rd['Y'][i].shape == (t, Y_shape[i])
            

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





 


 
 


if __name__ == "__main__":     
 
    all_videos = glob.glob("/home/richard/Downloads/TData1/*.hevc")
 

    n_workers = 8
    video_batchs = [[] for i in range(n_workers)]

    for i in range(len(all_videos)):
        video_batchs[i % n_workers].append(all_videos[i])
    
    
    print(f'load dists: {[len(video_batch) for video_batch in video_batchs]}')
    
    VecEnv(n_workers-1, video_batchs[:-1])
    
    
    worker(n_workers-1, None, None, video_batchs[n_workers-1])
        
 
    count = 0
    for vi in all_videos:
        data_file = vi.replace("fcamera.hevc", "data.pkl")
        assert os.path.isfile(data_file), data_file
        count += 1
    print(f'all data file verified: {count}')
