'''   YJW, HC, JLL, 2021.8.14 - 2024.2.22
from /home/richard/YPN/Leon/main.py
     /home/richard/OP079C2/selfdrive/modeld/models/driving079.cc

fcamera.hevc: vanishing point adjustments (in lanes_image_space, parserB6)
  StartPt, PATH_DISTANCE = 3, 192
  W/2. + 29, H/2. - 40, height = 1.2, path: + 0,   lll: - 0.2, rll: - 0.7
video.hevc:
  StartPt, PATH_DISTANCE = 4, 192
  W/2. + 10, H/2. - 58, height = 1.4, path: + 0.1, lll: + 0.1, rll: - 0.5

(sconsvenv) richard@Liu:~/openpilot/aJLL/Model$ python simulatorB6.py
Input:
  /home/richard/openpilot/aJLL/Model/saved_model/B6.keras
  /home/richard/dataB6/UHD--2018-08-02--08-34-47--32/video.hevc                # USA video (Rav4)
  /home/.../dataC/8bfda98c9c9e4291|2020-05-11--03-00-57--61/fcamera.hevc    # Taiwan video (Prius)
    B6.keras imitates supercombo079.keras and predicts driving path, lane lines, etc. on fcamera.hevc
    parserB6.py parses 12 outputs from B6.keras and supercombo079.keras on fcamera.hevc
Output:
  /home/richard/openpilot/aJLL/Model/output/B6Sim.png
  sim_output.txt, sim_output0_11.txt
'''
import os
import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model

from common.transformations.model import medmodel_intrinsics
from lanes_image_space import transform_points
from cameraB3 import transform_img, eon_intrinsics
from parserB6 import parser
from modelB6 import get_model
# camerafile = '/home/richard/dataB6/UHD--2018-08-02--08-34-47--32/video.hevc'
camerafile = '/home/richard/dataB6/UHD--2018-08-02--08-34-47--33/video.hevc'
# camerafile = '/home/richard/dataB6/UHD--2018-08-02--08-34-47--37/video.hevc'
# camerafile = '/home/richard/openpilot/tools/replay/dataC/8bfda98c9c9e4291|2020-05-11--03-00-57/61/fcamera.hevc'

# supercombo = load_model('saved_model/B6.keras', compile = False)   # 1 out = (1, 2383)
#supercombo = load_model('models/supercombo079.keras', compile = False)   # 12 outs
#print(supercombo.summary())





supercombo = get_model()
supercombo.load_weights(f'ckpt/modelB6-{700}.h5')  # for retraining

# supercombo = load_model('saved_model/supercombo079.keras', compile=False)




'''
supercombo = load_model('saved_model/B6.keras', compile = False)   # 1 out = (1, 2383)
  99 :  new_x_path = [567.7336717867292, 625.5671301933083, 552.933855447142] parsed["path"][0] = [ 0.23713899  0.16713709 -0.5016851 ]
supercombo = load_model('models/supercombo079.keras', compile = False)   # 12 outs
  99 :  new_x_path = [699.0010597977995, 666.3408217211254, 646.5449165835355] parsed["path"][0] = [-0.3373536 -0.3910733 -0.4009965]
'''
PATH_DISTANCE = 192
LANE_OFFSET = 1.8
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

class NumpyEncoder(json.JSONEncoder):  # encodes numpy-array obj to json format
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def plot_label(frame_no, x_left, y_left, x_path, y_path, x_right, y_right):
  window_name = 'Frame # ' + str(frame_no)
  pic = np.zeros((874, 1164, 3), dtype=np.uint8)
  cv2.line(pic, (int(x_left[0]), int(y_left[0])), (int(x_left[-1]), int(y_left[-1])), (255,255,255), 5)
  cv2.line(pic, (int(x_path[0]), int(y_path[0])), (int(x_path[-1]), int(y_path[-1])), (255,255,255), 5)
  cv2.line(pic, (int(x_right[0]), int(y_right[0])), (int(x_right[-1]), int(y_right[-1])), (255,255,255), 5)
  cv2.imshow(window_name, pic)
  cv2.waitKey(1000)
  input("Press ENTER to close Frame # ...")
  if cv2.waitKey(1000) == 27:   # if ENTER is pressed
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()

    #cv2.imwrite('output.png', pic)
'''
bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=>  CbYUV (6, 291, 582) = (C, H, W) [key: 1311 =  874x3/2]
sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=>  CsYUV (6, 128, 256) = (C, H, W) [key:  384 =  256x3/2]
'''


def sYUVs_to_CsYUVs(sYUVs):   # see hevc2yuvh5.py and main.py
    #--- sYUVs.shape = (2, 384, 512)
  H = (sYUVs.shape[1]*2)//3   # = 384x2//3 = 256
  W = sYUVs.shape[2]
  CsYUVs = np.zeros((sYUVs.shape[0], 6, H//2, W//2), dtype=np.uint8)

  CsYUVs[:, 0] = sYUVs[:, 0:H:2, 0::2]
  CsYUVs[:, 1] = sYUVs[:, 1:H:2, 0::2]
  CsYUVs[:, 2] = sYUVs[:, 0:H:2, 1::2]
  CsYUVs[:, 3] = sYUVs[:, 1:H:2, 1::2]
  CsYUVs[:, 4] = sYUVs[:, H:H+H//4].reshape((-1, H//2,W//2))
  CsYUVs[:, 5] = sYUVs[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  CsYUVs = np.array(CsYUVs).astype(np.float32)

  return CsYUVs  #--- CsYUVs.shape = (2, 6, 128, 256)



sYUVs = np.zeros((2, 384, 512), dtype='float32')
desire = np.zeros((1, 8), dtype='float32')
traffic_convection = np.zeros((1, 2), dtype='float32')
state = np.zeros((1, 512), dtype='float32')

# desire[:, 0] = 1.0
# traffic_convection[:, 0] = 1.0
 
cap = cv2.VideoCapture(camerafile)

x_lspace = np.linspace(1, PATH_DISTANCE, PATH_DISTANCE)   # linear spacing: linspace(start, stop, num), num: total number of items (pionts)

(ret, previous_frame) = cap.read()   # read 1st frame and set it to previous_frame

if not ret:
   exit()
else:
  frame_no = 1
  bYUV = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YUV_I420)   # from big BGR to big YUV
  sYUVs[0] = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                           output_size=(512, 256))   # resize bYUVs to small YUVs
    #--- sYUVs.shape = (2, 384, 512)


fig = plt.figure('OPNet Simulator')


#while True:
# for i in range(3):
for i in range(1200):
  (ret, current_frame) = cap.read()
  if not ret: break
  frame_no += 1

  frame = current_frame.copy()
  bYUV = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV_I420) # shape: (1311, 1164).

 

  sYUVs[1] = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                           output_size=(512, 256))


        # sYUVs = RGB_to_sYUVs(cap, frame_count)
        # CsYUVs = sYUVs_to_CsYUVs(sYUVs)
        # for i in range(frame_count):
        #   h5f['X'][i] = CsYUVs[i]

  if frame_no > 1:
    print("#--- frame_no =", frame_no)
    CsYUVs = sYUVs_to_CsYUVs(sYUVs)
    # print(f'## np.vstack(CsYUVs[0:2])[None].shape: {np.vstack(CsYUVs[0:2])[None].shape}')
    
    inputs = [np.vstack(CsYUVs[0:2])[None], desire, traffic_convection, state]

    # outputs = supercombo(inputs)
    # outputs = [a.numpy() for a in outputs]
    outputs = supercombo.predict(inputs)
    # print(f'outputs.shape: {len(outputs)}')
  
    if len(outputs) == 1:   # for B6.keras
      o0  = outputs[:, PATH_IDX:   LL_IDX]   #--- o0.shape = (1, 385)
      o1  = outputs[:, LL_IDX:     RL_IDX]
      o2  = outputs[:, RL_IDX:     LEAD_IDX]
      o3  = outputs[:, LEAD_IDX:   LONG_X_IDX]
      o4  = outputs[:, LONG_X_IDX: LONG_V_IDX]
      o5  = outputs[:, LONG_V_IDX: LONG_A_IDX]
      o6  = outputs[:, LONG_A_IDX: DESIRE_IDX]
      o7  = outputs[:, DESIRE_IDX: META_IDX]
      o8  = outputs[:, META_IDX:   PRED_IDX]
      o9  = outputs[:, PRED_IDX:   POSE_IDX]
      o10 = outputs[:, POSE_IDX:   STATE_IDX]
      o11 = outputs[:, STATE_IDX:  OUTPUT_IDX]
      outs = [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11]
    else:   # for supercombo079.keras
      outs = outputs

    parsed = parser(outs)
      #--- len(parsed) = 25
      #[print("#--- parsed[", x, "].shape =", parsed[x].shape) for x in parsed]   # see output.txt
    state = outs[-1]   # Important to refeed the state
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # cv2 reads images in BGR format (instead of RGB)

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
    new_x_left, new_y_left = transform_points(x_lspace, parsed["lll"][0])
    new_x_path, new_y_path = transform_points(x_lspace, parsed["path"][0])
    new_x_right, new_y_right = transform_points(x_lspace, parsed["rll"][0])

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

    if(i%5==0):
        plt.savefig(f'output/sim/sim-{i}.png')
    # plt.show()
    #   #plt.legend(['lll', 'rll', 'path'])
    # plt.pause(0.001)
    # input("Press ENTER to close ...")
    # # if cv2.waitKey(1000) == 27:   # if ENTER is pressed
    # cv2.destroyAllWindows()

    # ''' plot parsed lines
    # plot_label(frame_no, new_x_left, new_y_left, new_x_path, new_y_path, new_x_right, new_y_right)
    # with open("y_true.json", "w") as f:  # use dump() to write parsed (numpy array object) into the json file f
    #   json.dump(parsed, f, cls=NumpyEncoder) '''

    # ''' plot large image for checking the vanishing point '''
    # plt.clf()   # clear figure
    # plt.xlim(0, 1164)
    # plt.ylim(874, 0)
    # plt.plot()
    # plt.title("Original Scene")
    # new_x_left, new_y_left = transform_points(x_lspace, parsed["lll"][0])
    # new_x_path, new_y_path = transform_points(x_lspace, parsed["path"][0])
    # new_x_right, new_y_right = transform_points(x_lspace, parsed["rll"][0])
    # plt.plot(new_x_left, new_y_left, label='transformed', color='r')
    # plt.plot(new_x_path, new_y_path, label='transformed', color='g')
    # plt.plot(new_x_right, new_y_right, label='transformed', color='b')
    # plt.imshow(frame)
    # plt.pause(0.001)
    # input("Press ENTER to close ...")
    # # if cv2.waitKey(1000) == 27:   # if ENTER is pressed
    # # cv2.destroyAllWindows()

  sYUVs[0] = sYUVs[1]

'''print('#--- parsed["path"][0]  =', parsed["path"][0])
  print('#--- parsed["path"][0][:3]  =', parsed["path"][0][:3])
  print('#--- parsed["path"][0][-3:] =', parsed["path"][0][-3:])
  print('#--- new_x_path[:3]         =', new_x_path[:3])
  print('#--- new_x_path[-3:]        =', new_x_path[-3:])
  print('#--- new_y_path[:3]         =', new_y_path[:3])
  print('#--- new_y_path[-3:]        =', new_y_path[-3:])'''
print("#--- len(new_x_path) =", len(new_x_path))
print("#--- len(outputs) =", len(outputs))

input("Press ENTER to exit ...")
if cv2.waitKey(1000) == 27:   # if ENTER is pressed
  cv2.destroyAllWindows()
plt.close()
