"""   YPL, YJW, JLL, 2021.9.8 - 2024.2.21
for 230826, 240124, 240207
from /home/jinn/openpilot/aJLLold/Model/train_modelB6b.py

1. Use the output of supercombo079.keras as ground truth data to train modelB6
2. Tasks: Path Prediction + Lane Detection + Lead Car Detection
   y_true[2383] = (Ytrue0, Ytrue1, Ytrue2, ..., Ytrue10, Ytrue11)
   y_pred[2383] = outs[0] + ... + outs[11] in sim_output.txt.

Input:
  /home/jinn/dataB6/UHD--2018-08-02--08-34-47--32/yuv.h5
  /home/jinn/dataB6/UHD--2018-08-02--08-34-47--33/yuv.h5
  /home/jinn/dataB6/UHD--2018-08-02--08-34-47--37/yuv.h5
Output:
  /home/jinn/openpilot/aJLL/Model/saved_model/B6.keras
  /home/jinn/openpilot/aJLL/Model/saved_model/B6BW.hdf5
  /home/jinn/openpilot/aJLL/Model/output/B6Loss.png

Train and Validate Model: Run on 3 terminals
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python serverB6.py --port 5557
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python serverB6.py --port 5558 --validation
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python train_modelB6.py --port 5557 --port_val 5558

Test Model Step 1: /output/B6Sim.png
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python simulatorB6.py
Test Model Step 2: /output/B6Y.png
  (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python train_modelB6.py

Important Hyper-Parameters (tune these for better results):
BATCH_SIZE, STEPS, EPOCHS, learning_rate, decay_steps, decay_rate, weight_decay, clipvalue

Set BATCH_SIZE, STEPS, EPOCHS in serverB6.py. Set larger BATCH_SIZE if you have more GPUs.
Epochs (JLL) != EPOCHS (Keras); Steps (JLL) != STEPS (Keras); Run datagen_debug() for these.
"""
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as KB
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

from modelB6 import get_model
from serverB6 import client_generator, BATCH_SIZE, STEPS, EPOCHS
from datagenB6 import datagen_test

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# a batch of data. e.g. Ximgs.shape[0] == batch size.
def get_data(hwm, host, port, model):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    Ximgs_rgb, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4,\
                           Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11 = tup

    # print(f'Ximgs_rgb: {Ximgs_rgb.shape}')

    

    # Xins  = [Ximgs_rgb, Xin1, Xin2, Xin3]  #  (imgs, traffic_convection, desire, rnn_state)
    # Ytrue = np.hstack((Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5,
    #                          Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11))

    Xins  = [*tup[:4]]
    Xins = [tf.convert_to_tensor(x, dtype=tf.float32) for x in Xins]

    Ytrue = np.hstack(tup[4:])
    Ytrue = tf.convert_to_tensor(Ytrue, dtype=tf.float32)


    # print(f'Ytrue: {Ytrue}')

    # exit()
    
 


    #   #--- Xins[0].shape = (16, 12, 128, 256)
    #   #--- Ytrue.shape = (16, 2383)
    #   # we need the following two lines, otherwise we always get #--- y_true.shape[0] = None
    # p = model.predict(x=Xins)
    # loss1 = custom_loss(Ytrue, p)
    
    yield Xins, Ytrue



def maxae(y_true, y_pred):
  return KB.max(KB.abs(y_pred - y_true), axis=-1)


def custom_loss(y_true, y_pred):
    #--- y_true.shape = (None, None)
    #--- y_pred.shape = (None, 2383)
    #--- y_true.shape = (16, 2383)
    #--- y_pred.shape = (16, 2383)
    #--- max of y_true, max index = 266.34155 (array([0]), array([769]))    # at y_true(0, 769); 769-385=384=outs[1][384] in sim_output.txt
    #--- min of y_true, min index = -139.88947 (array([0]), array([1168]))  # MDN_GROUP_SIZE*LEAD_MDN_N+SELECTION=11*5+3=58
    # 1168-385-386-386=1168-1157=11=outs[3][11] (1st of 11, 2nd of 5); -139.889=data[1*11] in driving079.cc; why -???
  loss_CS = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)  # MC4 (Md Case 4)

  # print(f'loss_CS.shape: {loss_CS.shape}')

  loss_MSE = tf.keras.losses.mse(y_true, y_pred)  # MC5

  # print(f'loss_MSE.shape: {loss_MSE.shape}')

  loss = 0.5 * loss_CS + 0.5 * loss_MSE  # MC5

  metric = maxae(y_true, y_pred)

  # print(f'metric.shape: {metric.shape}')

  # exit()
  return loss, metric






# def maxae(y_true, y_pred):
#   return tf.math.maximum(tf.math.abs(y_pred - y_true), axis=-1)






class PrintLearningRate(Callback):
  def on_epoch_begin(self, epoch, logs={}):
    lr = KB.eval(self.model.optimizer.lr)
    print('\n', "epoch:", epoch, ', LR: {:.3g}'.format(lr))


def scheduler(epoch):
  return KB.get_value(model.optimizer.lr)

# global_epoch increment by 1 every STEPS number of data.
def lr_CosineDecayWarmup(global_epoch, total_epoches, warmup_epoches, hold, target_lr=1e-3):
  if global_epoch < warmup_epoches:
    learning_rate = target_lr * (global_epoch / warmup_epoches)  # linear warmup from 0 to target_lr
  elif global_epoch < warmup_epoches + hold:
    learning_rate = target_lr
  else:

    learning_rate = 0.5 * target_lr * ( 1 + np.cos( np.pi * float(global_epoch / total_epoches) ) )
  return learning_rate



class CosineDecayWarmup(Callback):

  def __init__(self, target_lr=1e-3, total_epoches=0, warmup_epoches=0, hold=0):
    super(CosineDecayWarmup, self).__init__()
    self.target_lr = target_lr
    self.total_epoches = total_epoches
    self.warmup_epoches = warmup_epoches
    self.hold = hold
    self.global_epoch = 0 # global_epoch increment by 1 every STEPS number of data.
    self.lrs = []

  def on_epoch_begin(self, epoch, logs=None):
    self.global_epoch = KB.get_value(epoch)

  def on_batch_begin(self, batch, logs=None):

    # global_epoch increment by 1 every STEPS number of data.
    lr = lr_CosineDecayWarmup(global_epoch=self.global_epoch, total_epoches=self.total_epoches,
                      warmup_epoches=self.warmup_epoches, hold=self.hold, target_lr=self.target_lr)
    #print("#--- self.global_epoch =", self.global_epoch)
    KB.set_value(self.model.optimizer.lr, lr)

  def on_batch_end(self, batch, logs=None):
    lr = model.optimizer.lr.numpy()
    self.lrs.append(lr)




if __name__=="__main__":
    start = time.time()
    AP = argparse.ArgumentParser(description='Training modelB6')
    AP.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    AP.add_argument('--port', type=int, default=5557, help='Port of server.')
    AP.add_argument('--port_val', type=int, default=5558, help='Port of server for validation dataset.')
    args = AP.parse_args()


    # Build model
    
    # img_shape = (12, 128, 256) # 2 yub img.
    img_shape = (256, 256, 3) # 2 yub img.

    desire_shape = (8,)
    traffic_convection_shape = (2,)
    rnn_state_shape = (512,)
    num_classes = 6
    model = get_model(img_shape, desire_shape, traffic_convection_shape, rnn_state_shape, num_classes)
    #model.summary()

    # Compile model
    filepath = "./saved_model/B6BW.hdf5"  # BW: Best Weights
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    total_epoches, warmup_epoches, hold = EPOCHS, 6, 4  # must: total_epoches > warmup_epoches + hold
    lr_schedule = CosineDecayWarmup(target_lr=1e-3, total_epoches=total_epoches, warmup_epoches=warmup_epoches, hold=hold)
    # optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0, weight_decay=0.004, clipvalue=1.0)

    printlr = PrintLearningRate()
    updatelr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint, printlr, updatelr, lr_schedule]

    # model.compile(optimizer=optimizer, loss=custom_loss, metrics=[maxae])


 






    model.load_weights(f'ckpt/modelB6-{90}.h5')  # for retraining

 
    train_dataset = get_data(20, args.host, port=args.port, model=model)
    val_dataset = get_data(20, args.host, port=args.port_val, model=model)


    def validate(n=25):
        
        print(f'## validating...')

        good = total = 0
        # steps = input_pipeline.get_dataset_info(dataset, 'test')['num_examples'] // batch_size

        # for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):

        losses = 0
        metrics = 0
        count = 0
        for i, batch in tqdm(enumerate(val_dataset)):
            if i >= n: break

            X, Y = batch
            count += len(Y)

            Y_pred = model(X, training=True) # (128, 10)
            loss, metric = train_loss_fn(Y, Y_pred)

            losses += loss.numpy().sum()
            metrics += metric.numpy().sum()

        assert count == n * BATCH_SIZE, f'{count} != {n * BATCH_SIZE}'

        losses = losses / count
        metrics = metrics / count

        print(f'validation - loss: {losses}, metric: {metrics}')

        return losses, metrics

 


    total_steps = 1000
    base_lr = 5e-4
    decay_type = 'cosine'
    warmup_steps = 5
    grad_norm_clip = 1
    linear_end=1e-5

    accum_steps = 1
    batch_size = BATCH_SIZE



    optimizer = tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=0.05, ema_momentum=0.9)        



    def lr_scheduler(step):

        lr = base_lr

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)

        if decay_type == 'linear':
            lr = linear_end + (lr - linear_end) * (1.0 - progress)
        elif decay_type == 'cosine':
            lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
        else:
            raise ValueError(f'Unknown lr type {decay_type}')

        if warmup_steps:
            lr = lr * np.minimum(1., step / warmup_steps)

        return np.asarray(lr, dtype=np.float32) 

    # train_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
    #                                     reduction=tf.keras.losses.Reduction.NONE)

    train_loss_fn = custom_loss


    @tf.function
    def _train_step(X_step, Y_step):

        # X_step, Y_step = batch_step

        with tf.GradientTape() as tape:
            Y_pred = model(X_step, training=True) # (128, 10)
            loss, metric = train_loss_fn(Y_step, Y_pred)
            loss = tf.reduce_mean(loss)
            metric = tf.reduce_mean(metric)

        grad = tape.gradient(loss, model.trainable_variables)
        #     prob = tf.nn.softmax(logits, axis=-1)
        #     loss = tf.reduce_mean(self.cce(Y_step, prob))

        # grad = tape.gradient(loss, self.model.trainable_variables)

        return loss, metric, grad
 


    log_interval = 5
    save_interval = 30
    validate_interval = 20

    update_counts = 0
    for epoch in range(100):

        # losses = []
        for i, batch in enumerate(train_dataset): # have ~600 training batches.

            X, Y = batch
                        
            if(len(X[0]) < batch_size):
                print(f'len(X) < batch_size: {len(X[0])}')
                break

            total_loss = 0.0
            total_metric = 0.0
            accum_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
            step_size = batch_size // accum_steps
            for step in range(accum_steps): 

                X_step = [a[step*step_size:(step+1)*step_size] for a in X]
                Y_step = Y[step*step_size:(step+1)*step_size]
 
                loss, metric, grad = _train_step(X_step, Y_step)

                for i in range(len(accum_gradients)):
                    accum_gradients[i] += grad[i]

                total_loss += loss.numpy() 
                total_metric += metric.numpy() 

            averaged_gradients = [accum_grad / tf.cast(accum_steps, tf.float32) for accum_grad in accum_gradients]
            clipped_grads, _ = tf.clip_by_global_norm(averaged_gradients, grad_norm_clip)
            optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))


            update_counts += 1
            lr_next = lr_scheduler(update_counts)
            tf.keras.backend.set_value(optimizer.learning_rate, lr_next)


            if update_counts % log_interval == 0:
                print(f'[{update_counts}] train_loss: {total_loss / accum_steps}' + \
                f', total_metric: {total_metric / accum_steps}' + f', lr: {lr_next}')


            if update_counts % validate_interval == 0:
                loss, metric = validate()                
            

            if update_counts % save_interval == 0:
                model.save_weights(f'ckpt/modelB6-{update_counts}.h5')

      
 







    exit()



 

    print('########### a')
    model.save('./saved_model/B6.keras')  # rename and save "long" trained .keras

    print('########### b')
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("#--- Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    print('########### c')
    # Plot training results
    plt.subplot(311)

    print('########### d')

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("loss")
    plt.legend(['train', 'validate'], loc='upper right')

    print('########### e')

    plt.subplot(312)
    plt.plot(history.history["maxae"])
    plt.plot(history.history["val_maxae"])
    plt.ylabel("maxae")
    plt.xlabel("epoch")
    plt.legend(['train', 'validate'], loc='upper right')

    plt.subplot(313)
    plt.plot(history.history["lr"])  # = lr (from Terminal) = lr_schedule (decayed)
    plt.ylabel("learning rate")

    plt.draw()
    plt.savefig('./output/B6Loss.png')
    plt.pause(0.5)
    input("Press ENTER to close ...")
    plt.close()

    ''' 2. Test model

    # Test model by only 2 images
    # Run: (sconsvenv) jinn@Liu:~/openpilot/aJLL/Model$ python train_modelB6.py
    print("#--- Testing ...")

    # Load pre-trained model for predicting
    #model.load_weights('./saved_model/B6.keras')
    model.load_weights('./saved_model/B6BW.hdf5')

    camera_file = '/home/jinn/dataB6/UHD--2018-08-02--08-34-47--33/video.hevc'
    Xins, Ytrue = datagen_test(1, camera_file)
    Ypred = model.predict(x=Xins)
      #--- Testing Time: 00:00:04.57

    plt.subplot(211)
    Ytrue_1 = Ytrue[0]
    Ypred_1 = Ypred[0]
    plt.plot(Ytrue_1)
    plt.plot(Ypred_1)
    plt.legend(['Ytrue', 'Ypred'], loc='upper left')
    #plt.title('Y-2383 by Huber Loss', fontweight='bold')
    plt.title('MC5 Y-2383', fontweight='bold')

    plt.subplot(212)
    Ytrue_2 = Ytrue[0][0:385]
    Ypred_2 = Ypred[0][0:385]
    plt.plot(Ytrue_2)
    plt.plot(Ypred_2)
    plt.legend(['Ytrue', 'Ypred'], loc='upper left')
    plt.title('Y-385', fontweight='bold')

    plt.draw()
    plt.savefig('./output/B6YCM.png')
    plt.pause(0.5)
    input("Press ENTER to close ...")
    plt.close() '''

    ''' 3. Test CosineDecayWarmup

    epoches = np.arange(0, 1000, 1)
    lrs = []
    total_epoches, warmup_epoches, hold = len(epoches), 100, 200
    for epoch in epoches:
      lrs.append(lr_CosineDecayWarmup(epoch, total_epoches=total_epoches, warmup_epoches=warmup_epoches, hold=hold))
    schedule = CosineDecayWarmup(target_lr=1e-3, total_epoches=total_epoches, warmup_epoches=warmup_epoches, hold=hold)
    plt.plot(lrs)
    plt.draw()
    plt.pause(0.5)
    input("Press ENTER to close ...")
    plt.close() '''
