 
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






  
supercombo = load_model('saved_model/supercombo079.keras', compile=False)  

def supercombo_y(Ximgs, Xin1, Xin2, Xin3):
      
  Ximgs = np.expand_dims(Ximgs, axis=0) # (1, 12, 128, 256)        
  Xin1 = np.expand_dims(Xin1, axis=0)
  Xin2 = np.expand_dims(Xin2, axis=0)
  Xin3 = np.expand_dims(Xin3, axis=0)
  inputs = [Ximgs, Xin1, Xin2, Xin3]

  outputs = supercombo(inputs)

  return outputs




# def RGB_to_sYUVs(video):
 
#   sYUVs = []

#   ret, frame = video.read()

#   while ret:

#     bYUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420) # (1311, 1164)
  
#     sYUV = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
#                              yuv=True, output_size=(512, 256))  # (384, 512)
    
    
#     sYUVs.append(sYUV[None])
    
#     ret, frame = video.read()
  
#   sYUVs = np.concatenate(sYUVs)
  
#   return sYUVs



def RGB_to_sYUVs(frame):
  
   
    bYUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420) # (1311, 1164)

    sYUV = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
                            yuv=True, output_size=(512, 256))  # (384, 512)
    
    return sYUV

        

# def sYUVs_to_CsYUVs(sYUVs):
#     H = (sYUVs.shape[1]*2)//3  # 384x2//3 = 256
#     W = sYUVs.shape[2]         # 512
#     CsYUVs = np.zeros((sYUVs.shape[0], 6, H//2, W//2), dtype=np.uint8)

#     CsYUVs[:, 0] = sYUVs[:, 0:H:2, 0::2]  # [2::2] get every even starting at 2
#     CsYUVs[:, 1] = sYUVs[:, 1:H:2, 0::2]  # [start:end:step], [2:4:2] get every even starting at 2 and ending at 4
#     CsYUVs[:, 2] = sYUVs[:, 0:H:2, 1::2]  # [1::2] get every odd index, [::2] get every even
#     CsYUVs[:, 3] = sYUVs[:, 1:H:2, 1::2]  # [::n] get every n-th item in the entire sequence
#     CsYUVs[:, 4] = sYUVs[:, H:H+H//4].reshape((-1, H//2, W//2))
#     CsYUVs[:, 5] = sYUVs[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))

 
#     return CsYUVs

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

    
    plt.savefig(f'output/gen/gen-{i}.png')  

    # exit()


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




if __name__ == "__main__":
    #all_dirs = os.listdir('/home/richard/dataB')
    all_dirs = os.listdir('/home/richard/dataB6')
    # makeYUV(all_dirs)



    all_videos = glob.glob("/home/richard/Downloads/TData1/*.hevc")
    # print(f'len(all_videos): {len(all_videos)}')

    # chunk_size = len(all_videos) // 8

    # makeYUV(all_videos[3:10])
    # getFrame_rgb()

    all_videos = all_videos[3:100]


    batch_size = 8
    n_batchs = len(all_videos) // batch_size
    
    # for vi in all_videos:
    for i in range(n_batchs):        

        batch = all_videos[i*batch_size:(i+1)*batch_size]

        # yuvh5 = vi.replace('video.hevc','yuv.h5')
        # frames_rgb = read_hevc(vi)

        # yuvh5 = vi.replace('fcamera.hevc','yuv.h5')
        # yuvh5 = yuvh5.replace('/TData1/','/TData1-pp/')

        output_files = [vi.replace('fcamera.hevc', 'data.pkl').replace('/TData1/','/TData1-pp/') for vi in batch]
        
        # yuvh5 = vi.replace('video.hevc','test.h5')
        # print("## video =", vi)
    
        caps = [cv2.VideoCapture(vi) for vi in batch]    

        rets, frames = zip(*[cap.read() for cap in caps])
        YUVs_next = [RGB_to_YUV(frame) for frame in frames] 

        rets, frames = zip(*[cap.read() for cap in caps])
        # if not np.array(rets).all(): continue
        # YUVs_next = [RGB_to_YUV(frame) for frame in frames] 

        rnn_sts = np.zeros((batch_size, 512))

        Xin3s = [[] for i in range(batch_size)]
        Ys = [[] for i in range(batch_size)]

        while np.array(rets).any():

            # sYUVs = [RGB_to_sYUVs(frame) for frame in frames] 
            # CsYUVs = [sYUVs_to_CsYUVs(sYUV) for sYUV in sYUVs]
            YUVs = YUVs_next
            # for i in range(batch_size):
            #     frame = frames[i]
            #     if(not rets[i]): frame = YUVs[i]
            #     YUVs_next = [RGB_to_YUV(frame) for frame in frames] 

            YUVs_next = [frames[i] if rets[i] else YUVs[i] for i in range(batch_size)]

            # vsX1 = yuvX[i]
            # vsX2 = yuvX[i+1]

            # # X[0][i] = frames_rgb[i]

            # # Ximgs[i] = np.vstack((vsX1, vsX2))  
            # Ximg = np.vstack((vsX1, vsX2)) 
 
            Ximgs = np.concatenate([np.vstack((YUVs[i], YUVs_next[i]))[None] for i in range(batch_size)])
            assert Ximgs.shape == (b, 12, 128, 256)
  
            outs = supercombo([Ximgs, np.zeros((batch_size, 8)), np.zeros((batch_size, 2)), rnn_sts])
  
            
            for i in range(batch_size):
                Xin3s[i].append(rnn_sts[i])
                Ys[i].append(outs[i])





            rnn_sts = outs[11][:] # (batch_size, 512)
                
            assert len(outs) == 12

            for j in range(len(outs)):
                Y[i] = outs[j]


            # outs = supercombo_y(Ximg, np.zeros((8, )), np.zeros((2, )), rnn_st)

            rets, frames = zip(*[cap.read() for cap in caps])


        for cap in caps: cap.release()










# import os
# import io
# import json
# import torch
# from math import pi
# import numpy as np
# from scipy.interpolate import interp1d
# import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
# from utils import warp, generate_random_params_for_warp
# from view_transform import calibration

# import utils_comma2k19.orientation as orient
# import utils_comma2k19.coordinates as coord




# class PlanningDataset(Dataset):
#     def __init__(self, root='data', json_path_pattern='p3_%s.json', split='train'):
#         self.samples = json.load(open(os.path.join(root, json_path_pattern % split)))
#         print('PlanningDataset: %d samples loaded from %s' % 
#               (len(self.samples), os.path.join(root, json_path_pattern % split)))
#         self.split = split

#         self.img_root = os.path.join(root, 'nuscenes')
#         self.transforms = transforms.Compose(
#             [
#                 # transforms.Resize((900 // 2, 1600 // 2)),
#                 # transforms.Resize((9 * 32, 16 * 32)),
#                 transforms.Resize((128, 256)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.3890, 0.3937, 0.3851],
#                                      [0.2172, 0.2141, 0.2209]),
#             ]
#         )

#         self.enable_aug = False
#         self.view_transform = False

#         self.use_memcache = False
#         if self.use_memcache:
#             self._init_mc_()

#     def _init_mc_(self):
#         from petrel_client.client import Client
#         self.client = Client('~/petreloss.conf')
#         print('======== Initializing Memcache: Success =======')

#     def _get_cv2_image(self, path):
#         if self.use_memcache:
#             img_bytes = self.client.get(str(path))
#             assert(img_bytes is not None)
#             img_mem_view = memoryview(img_bytes)
#             img_array = np.frombuffer(img_mem_view, np.uint8)
#             return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#         else:
#             return cv2.imread(path)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         imgs, future_poses = sample['imgs'], sample['future_poses']

#         # process future_poses
#         future_poses = torch.tensor(future_poses)
#         future_poses[:, 0] = future_poses[:, 0].clamp(1e-2, )  # the car will never go backward

#         imgs = list(self._get_cv2_image(os.path.join(self.img_root, p)) for p in imgs)
#         imgs = list(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs)  # RGB

#         # process images
#         if self.enable_aug and self.split == 'train':
#             # data augumentation when training
#             # random distort (warp)
#             w_offsets, h_offsets = generate_random_params_for_warp(imgs[0], random_rate=0.1)
#             imgs = list(warp(img, w_offsets, h_offsets) for img in imgs)

#             # random flip
#             if np.random.rand() > 0.5:
#                 imgs = list(img[:, ::-1, :] for img in imgs)
#                 future_poses[:, 1] *= -1
            

#         if self.view_transform:
#             camera_rotation_matrix = np.linalg.inv(np.array(sample["camera_rotation_matrix_inv"]))
#             camera_translation = -np.array(sample["camera_translation_inv"])
#             camera_extrinsic = np.vstack((np.hstack((camera_rotation_matrix, camera_translation.reshape((3, 1)))), np.array([0, 0, 0, 1])))
#             camera_extrinsic = np.linalg.inv(camera_extrinsic)
#             warp_matrix = calibration(camera_extrinsic, np.array(sample["camera_intrinsic"]))
#             imgs = list(cv2.warpPerspective(src = img, M = warp_matrix, dsize= (256,128), flags= cv2.WARP_INVERSE_MAP) for img in imgs)

#         # cvt back to PIL images
#         # cv2.imshow('0', imgs[0])
#         # cv2.imshow('1', imgs[1])
#         # cv2.waitKey(0)
#         imgs = list(Image.fromarray(img) for img in imgs)
#         imgs = list(self.transforms(img) for img in imgs)
#         input_img = torch.cat(imgs, dim=0)

#         return dict(
#             input_img=input_img,
#             future_poses=future_poses,
#             camera_intrinsic=torch.tensor(sample['camera_intrinsic']),
#             camera_extrinsic=torch.tensor(sample['camera_extrinsic']),
#             camera_translation_inv=torch.tensor(sample['camera_translation_inv']),
#             camera_rotation_matrix_inv=torch.tensor(sample['camera_rotation_matrix_inv']),
#         )




# class SequencePlanningDataset(PlanningDataset):
#     def __init__(self, root='data', json_path_pattern='p3_%s.json', split='train'):
#         print('Sequence', end='')
#         self.fix_seq_length = 18
#         super().__init__(root=root, json_path_pattern=json_path_pattern, split=split)

#     def __getitem__(self, idx):
#         seq_samples = self.samples[idx]
#         seq_length = len(seq_samples)
#         if seq_length < self.fix_seq_length:
#             # Only 1 sample < 28 (==21)
#             return self.__getitem__(np.random.randint(0, len(self.samples)))
#         if seq_length > self.fix_seq_length:
#             seq_length_delta = seq_length - self.fix_seq_length
#             seq_length_delta = np.random.randint(0, seq_length_delta+1)
#             seq_samples = seq_samples[seq_length_delta:self.fix_seq_length+seq_length_delta]

#         seq_future_poses = list(smp['future_poses'] for smp in seq_samples)
#         seq_imgs = list(smp['imgs'] for smp in seq_samples)

#         seq_input_img = []
#         for imgs in seq_imgs:
#             imgs = list(self._get_cv2_image(os.path.join(self.img_root, p)) for p in imgs)
#             imgs = list(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs)  # RGB
#             imgs = list(Image.fromarray(img) for img in imgs)
#             imgs = list(self.transforms(img) for img in imgs)
#             input_img = torch.cat(imgs, dim=0)
#             seq_input_img.append(input_img[None])
#         seq_input_img = torch.cat(seq_input_img)

#         return dict(
#             seq_input_img=seq_input_img,  # torch.Size([28, 10, 3])
#             seq_future_poses=torch.tensor(seq_future_poses),  # torch.Size([28, 6, 128, 256])
#             camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']),
#             camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']),
#             camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']),
#             camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']),
#         )





# # val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt',
# #                                 'data/comma2k19/', 'demo', use_memcache=False, return_origin=True)
# class Comma2k19SequenceDataset(PlanningDataset):

#     def __init__(self, split_txt_path, prefix, mode, use_memcache=True, return_origin=False):

#         self.split_txt_path = split_txt_path
#         self.prefix = prefix

#         self.samples = open(split_txt_path).readlines()
#         self.samples = [i.strip() for i in self.samples]

#         assert mode in ('train', 'val', 'demo')

#         self.mode = mode

#         if self.mode == 'demo':
#             print('Comma2k19SequenceDataset: DEMO mode is on.')

#         self.fix_seq_length = 800 if mode == 'train' else 800

#         self.transforms = transforms.Compose(
#             [
#                 # transforms.Resize((900 // 2, 1600 // 2)),
#                 # transforms.Resize((9 * 32, 16 * 32)),
#                 transforms.Resize((128, 256)), # (h, w)
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.3890, 0.3937, 0.3851], # mean
#                                      [0.2172, 0.2141, 0.2209] # std
#                                      ),
#             ]
#         )


#         # FULL_FRAME_SIZE = (1164, 874)
#         # W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
#         # eon_focal_length = FOCAL = 910.0

#         # # aka 'K' aka camera_frame_from_view_frame
#         # eon_intrinsics = np.array([
#         #   [FOCAL,   0.,   W/2.],
#         #   [  0.,  FOCAL,  H/2.],
#         #   [  0.,    0.,     1.]])

#         self.warp_matrix = calibration(
#             # go from car(road) frame (x->forward, y->left, z->up) to 
#                 # camera frame (x->right, y->down, z->forward) (with car frame's origin
#                 # lies 1.22 meters below that of camera frame)?
#             extrinsic_matrix=np.array([[ 0, -1,  0,    0],
#                                         [ 0,  0, -1, 1.22],
#                                         [ 1,  0,  0,    0],
#                                         [ 0,  0,  0,    1]]),
#             cam_intrinsics=np.array([[910, 0, 582],
#                                     [0, 910, 437],
#                                     [0,   0,   1]]),
            
#             # [x, y, z] -> [x, -y, 1.22 - z].
#             device_frame_from_road_frame=np.hstack((np.diag([1, -1, -1]), [[0], [0], [1.22]])))


#         self.use_memcache = use_memcache
#         if self.use_memcache:
#             self._init_mc_()

#         self.return_origin = return_origin

#         # from OpenPilot
#         self.num_pts = 10 * 20  # 10 s * 20 Hz = 200 frames
#         self.t_anchors = np.array(
#             (0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
#              0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
#              0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
#              2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
#              3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
#              6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
#              8.7890625 ,  9.38476562, 10.)
#         )
#         self.t_idx = np.linspace(0, 10, num=self.num_pts)


#     def _get_cv2_vid(self, path):
#         if self.use_memcache:
#             path = self.client.generate_presigned_url(str(path), client_method='get_object', expires_in=3600)
#         return cv2.VideoCapture(path)


#     def _get_numpy(self, path):
#         if self.use_memcache:
#             bytes = io.BytesIO(memoryview(self.client.get(str(path))))
#             return np.lib.format.read_array(bytes)
#         else:
#             return np.load(path)



#     def __getitem__(self, idx): # each item is all frames of an video (.hevc).

#         seq_sample_path = self.prefix + self.samples[idx]
#         cap = self._get_cv2_vid(seq_sample_path + '/video.hevc')

#         if (cap.isOpened() == False):
#             raise RuntimeError
        
#         imgs = []  # <--- all frames here
#         origin_imgs = []

#         while (cap.isOpened()):
#             ret, frame = cap.read()
#             if ret == True:
#                 imgs.append(frame)
#                 if self.return_origin: origin_imgs.append(frame)
#             else: break

#         cap.release()

#         seq_length = len(imgs)

#         if self.mode == 'demo':
#             # num_pts: 200.
#             self.fix_seq_length = seq_length - self.num_pts - 1


#         if seq_length < self.fix_seq_length + self.num_pts:
#             print('The length of sequence', seq_sample_path, 'is too short',
#                   '(%d < %d)' % (seq_length, self.fix_seq_length + self.num_pts))
#             return self.__getitem__(idx+1)


#         seq_length_delta = seq_length - (self.fix_seq_length + self.num_pts) # 1 for 'demo'.
#         seq_length_delta = np.random.randint(1, seq_length_delta+1) # low=1, high=2.



#         seq_start_idx = seq_length_delta
#         seq_end_idx = seq_length_delta + self.fix_seq_length



#         # seq_input_img
#         imgs = imgs[seq_start_idx-1: seq_end_idx]  # contains one more img
#         imgs = [cv2.warpPerspective(src=img, M=self.warp_matrix, dsize=(512,256),   
#                                          flags=cv2.WARP_INVERSE_MAP) for img in imgs]
        
#         imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
#         imgs = list(Image.fromarray(img) for img in imgs)

#         # resize from (512, 256) to (256, 128) and normalize.
#         imgs = list(self.transforms(img)[None] for img in imgs)

#         input_img = torch.cat(imgs, dim=0)  # [N+1, 3, H, W]

#         del imgs

#         # stack two consecutive images img[t] & img[t+1].
#         input_img = torch.cat((input_img[:-1, ...], input_img[1:, ...]), dim=1)

#         # positions are in ECEF (earth-centric) frame in meters.
#         frame_positions = self._get_numpy(
#                             self.prefix + self.samples[idx] + '/global_pose/frame_positions'
#                     )[seq_start_idx: seq_end_idx + self.num_pts]
        
#         # rotate ECEF frame into local frame.
#         frame_orientations = self._get_numpy(
#                             self.prefix + self.samples[idx] + '/global_pose/frame_orientations'
#                     )[seq_start_idx: seq_end_idx + self.num_pts]


#         # future_poses: future car positions (start from frame i to frame i+num_pts(=200)) relative to car's local coord sys.
#         future_poses = []
#         for i in range(self.fix_seq_length):

#             # quaternion to rotation matrix.
#             ecef_from_local = orient.rot_from_quat(frame_orientations[i])

#             local_from_ecef = ecef_from_local.T
#             frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, 
#                                     frame_positions - frame_positions[i]).astype(np.float32)

#             # Time-Anchor like OpenPilot
#             fs = [interp1d(self.t_idx, frame_positions_local[i: i+self.num_pts, j]) for j in range(3)]
#             interp_positions = [fs[j](self.t_anchors)[:, None] for j in range(3)]
#             interp_positions = np.concatenate(interp_positions, axis=1)
            
#             # relative to local coord sys and start from frame i to frame i+num_pts.
#             future_poses.append(interp_positions)
            
#         future_poses = torch.tensor(np.array(future_poses), dtype=torch.float32)

#         rtn_dict = dict(
#             seq_input_img=input_img,  # torch.Size([N, 6, 128, 256])
#             seq_future_poses=future_poses,  # torch.Size([N, num_pts, 3])
#             # camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']),
#             # camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']),
#             # camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']),
#             # camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']),
#         )

#         # For DEMO
#         if self.return_origin:
#             origin_imgs = origin_imgs[seq_start_idx: seq_end_idx]
#             origin_imgs = [torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None] for img in origin_imgs]
#             origin_imgs = torch.cat(origin_imgs, dim=0)  # N, H_ori, W_ori, 3
#             rtn_dict['origin_imgs'] = origin_imgs

#         return rtn_dict
