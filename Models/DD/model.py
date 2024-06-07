

import tensorflow as tf
from EfficientNet import EfficientNetB2

 

 

class SequencePlanningNetwork(tf.keras.Model):
    def __init__(self, M, num_pts):
        super().__init__()
      
        self.M = M # 5.
        
        num_pts = 192
        self.num_pts = num_pts     
        
            

        '''EfficientNetB2
            in shape: (b, 12, 128, 256) .
            out shape: (b, 4, 8, 1408).
            ch format: last.
        '''        
        self.backbone = EfficientNetB2()
 

        self.plan_head = tf.keras.Sequential([
            # 6, 450, 800 -> 1408, 14, 25
            # nn.AdaptiveMaxPool2d((4, 8)),  # 1408, 4, 8            
            tf.keras.layers.BatchNormalization(axis=3), # 4, 8, 1408
            tf.keras.layers.Conv2D(32, 1, padding='valid'),  # (b, 4, 8, 32).
            tf.keras.layers.BatchNormalization(axis=3), # (b, 4, 8, 32).
            tf.keras.layers.Flatten(), # (b, 1024).
            tf.keras.layers.ELU(),
        ]) # (b, 1024).
  
        # self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_state=True, return_sequences=True))

        self.gru = tf.keras.layers.GRU(512, return_state=True, return_sequences=True)
 
        self.plan_head_tip = tf.keras.Sequential([
            tf.keras.layers.Flatten(),  # (b, 1024).
            tf.keras.layers.ELU(),
            # tf.keras.layers.Dense(4096),  # (b, 4096).
            tf.keras.layers.Dense(256, activation='relu'),  # (b, 256).
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'), 
            tf.keras.layers.Dense(num_pts * 2 + 1) 
        ]) 
 

    def call(self, x, hidden, training=True):
        '''
            x: (b, 12, 128, 256).
        '''

        features = self.backbone(x, training=training) # (b, 4, 8, 1408).

        raw_preds = self.plan_head(features, training=training) # (b, 1024).


        # hidden_fw, hidden_bw = hidden # (b, 512), (b, 512).

        # print(f'raw_preds[:, None, :].shape: {raw_preds[:, None, :].shape}, hidden.shape: {hidden.shape}')
        raw_preds, hidden = self.gru(raw_preds[:, None, :], initial_state=hidden, training=training) # (b, 1, 512), (b, 512).

        raw_preds = self.plan_head_tip(raw_preds, training=training) # (b, 2*num_pts+1).

 
        return raw_preds, hidden # (b, 2*num_pts+1), .

 






class MultipleTrajectoryPredictionLoss(tf.keras.Model):

    def __init__(self, alpha, M, num_pts, distance_type='angle'):
        
        super().__init__()
  
        self.num_pts = num_pts # 192.
           
        self.reg_loss = tf.keras.losses.Huber(delta=1.0, reduction='none')

      
 
    def call(self, traj_pred, traj_true, training=True):
        ''' 
            traj_pred: (b, 2*num_pts+1), 
            gt: (b, 2*num_pts+1).
        '''        

        path_true = traj_true[:, :self.num_pts] # (b, num_pts).
        valid_len_true = tf.clip_by_value(traj_true[:, 2*self.num_pts], 5, self.num_pts-1) # (b,).
        
        L = tf.cast(valid_len_true, dtype=tf.int32) # (b,).

        path_pred = traj_pred[:, :self.num_pts] # (b, num_pts).
        valid_len_pred = tf.clip_by_value(traj_pred[:, 2*self.num_pts], 5, self.num_pts-1) # (b,).
        
        valid_len_loss = self.reg_loss(valid_len_true, valid_len_pred) # (,).
 
        reg_loss = self.reg_loss(path_true, path_pred) # (b,). 
        reg_loss = tf.math.reduce_mean(reg_loss, axis=0) # (,).
        
        return reg_loss, valid_len_loss # (,), (,).

