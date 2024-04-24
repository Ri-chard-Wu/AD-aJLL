"""   JLL  2024.2.7
for 240207
from A. Rosebrock, Keras: Starting, stopping, and resuming training
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.nn.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import argparse
import cv2
import sys
import os

from common.transformations.model import medmodel_intrinsics
from lanes_image_space import transform_points
from cameraB3 import transform_img, eon_intrinsics
from parserB6 import parser
def plot_label(frame_no, x_left, y_left, x_path, y_path, x_right, y_right):
def sYUVs_to_CsYUVs(sYUVs):
sYUVs = np.zeros((2, 384, 512), dtype=np.uint8)
desire = np.zeros((1, 8))
traffic_convection = np.zeros((1, 2))
state = np.zeros((1, 512))
cap = cv2.VideoCapture(camerafile)
x_lspace = np.linspace(1, PATH_DISTANCE, PATH_DISTANCE)
  bYUV = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YUV_I420)
  sYUVs[0] = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                           output_size=(512, 256))
for i in range(3):
  (ret, current_frame) = cap.read()
  frame = current_frame.copy()
  bYUV = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV_I420)
  sYUVs[1] = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                           output_size=(512, 256))
    CsYUVs = sYUVs_to_CsYUVs(sYUVs)
    inputs = [np.vstack(CsYUVs[0:2])[None], desire, traffic_convection, state]
    outputs = supercombo.predict(inputs)
    if len(outputs) == 1:   # for B6.keras
      o0  = outputs[:, PATH_IDX:   LL_IDX]
			...
      o11 = outputs[:, STATE_IDX:  OUTPUT_IDX]
      outs = [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11]
    else:   # for supercombo079.keras
      outs = outputs
    parsed = parser(outs)
    plt.title("Overlay Scene")
    new_x_left, new_y_left = transform_points(x_lspace, parsed["lll"][0])
    new_x_path, new_y_path = transform_points(x_lspace, parsed["path"][0])



# study of patience for the learning rate drop schedule on the blobs problem
from sklearn.datasets import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras import backend
from matplotlib import pyplot

# monitor the learning rate
class LearningRateMonitor(Callback):
 # start of training
 def on_train_begin(self, logs={}):
 self.lrates = list()

 # end of each training epoch
 def on_epoch_end(self, epoch, logs={}):
 # get and store the learning rate
 optimizer = self.model.optimizer
 lrate = float(backend.get_value(self.model.optimizer.lr))
 self.lrates.append(lrate)

# prepare train and test dataset
def prepare_data():
 # generate 2d classification dataset
 X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
 # one hot encode output variable
 y = to_categorical(y)
 # split into train and test
 n_train = 500
 trainX, testX = X[:n_train, :], X[n_train:, :]
 trainy, testy = y[:n_train], y[n_train:]
 return trainX, trainy, testX, testy

# fit a model and plot learning curve
def fit_model(trainX, trainy, testX, testy, patience):
 # define model
 model = Sequential()
 model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 model.add(Dense(3, activation='softmax'))
 # compile model
 opt = SGD(lr=0.01)
 model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
 # fit model
 rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, min_delta=1E-7)
 lrm = LearningRateMonitor()
 history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0, callbacks=[rlrp, lrm])
 return lrm.lrates, history.history['loss'], history.history['accuracy']

# create line plots for a series
def line_plots(patiences, series):
 for i in range(len(patiences)):
 pyplot.subplot(220 + (i+1))
 pyplot.plot(series[i])
 pyplot.title('patience='+str(patiences[i]), pad=-80)
 pyplot.show()

# prepare dataset
trainX, trainy, testX, testy = prepare_data()
# create learning curves for different patiences
patiences = [2, 5, 10, 15]
lr_list, loss_list, acc_list, = list(), list(), list()
for i in range(len(patiences)):
 # fit model and plot learning curves for a patience
 lr, loss, acc = fit_model(trainX, trainy, testX, testy, patiences[i])
 lr_list.append(lr)
 loss_list.append(loss)
 acc_list.append(acc)
# plot learning rates
line_plots(patiences, lr_list)
# plot loss
line_plots(patiences, loss_list)
# plot accuracy
line_plots(patiences, acc_list)
