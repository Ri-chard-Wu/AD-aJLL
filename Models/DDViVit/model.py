

import tensorflow as tf
from EfficientNet import EfficientNetB2
from FastVit import FastViT
from TransformerEncoder import TransformerEncoder
 
from parameter import transformerEncoder_args as enc_args
from parameter import AttrDict



def tanh_rescale(x, _min, _max):
    return 0.5*(x+1.0) * (_max - _min) + _min



class SequencePlanningNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
      
        num_pts = 192
        self.num_pts = num_pts     
        
        '''EfficientNetB2
            in shape: (b, 128, 256, 12) .
            out shape: (b, 1024). 
        '''        
        # self.backbone = EfficientNetB2()
 
        self.backbone = FastViT(
                AttrDict({
                    'layers': [2, 2, 6, 2],
                    'embed_dims': [64, 128, 256, 512],
                    'token_mixers': ("repmixer", "repmixer", "repmixer", "repmixer"),
                    'pos_embs': [None, None, None, None],
                    'mlp_ratios': [3, 3, 3, 3],
                    'downsamples': [True, True, True, True]
                })
            )
        self.backbone.load_ckpt('ckpt', 'fastvit-acc0p96.pkl')  
  
        # # for i, layer in enumerate(self.backbone.layers):
        # #     if layer.name == 'yuv2rgb': continue
        # #     layer.trainable = False
        # #     print(f'[{i}] {layer.name}: layer.trainable = False')
     



        # self.backbone = tf.keras.Sequential([ 
        #             tf.keras.layers.Conv2D(filters=64, kernel_size=8, strides=5),
        #             tf.keras.layers.ReLU(),
        #             tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=3),
        #             tf.keras.layers.ReLU(), 
        #             tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1),
        #             tf.keras.layers.ReLU(),
        #             tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1),
        #             tf.keras.layers.ReLU(),
        #             tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1),
        #             tf.keras.layers.ReLU(),
        #             tf.keras.layers.Flatten(),
        #             tf.keras.layers.Dense(1024),
        #             tf.keras.layers.ReLU()
        #         ])


  

        self.plan_head = tf.keras.Sequential([ 
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(enc_args.hidden_size), # (b, hid_sz).
            tf.keras.layers.ReLU(),
        ]) # (b, hid_sz).


        self.plan_head_tip = tf.keras.Sequential([
            tf.keras.layers.Flatten(), 
            # tf.keras.layers.ELU(),  # (b, 768).
            tf.keras.layers.Dense(256, activation='relu'),  # (b, 256).
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'), 
            # tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2*num_pts+1),
        ]) 
 
        # self.feature_sequence = deque(maxlen=100) # most recent 100 frames as input to transformer encoder.
 
        self.transformerEncoder = TransformerEncoder()
        
        # self.cls_token = self.add_weight("cls",
        #                                  shape=[1,
        #                                         1,
        #                                         enc_args.hidden_size],
        #                                  initializer=tf.keras.initializers.RandomNormal(),
        #                                  dtype=tf.float32)

        # self.pos_embedding = self.add_weight("Transformer/posembed_input/pos_embedding",
        #                                      shape=[1, enc_args.seq_len + 1, enc_args.hidden_size],
        #                                      initializer=tf.keras.initializers.RandomNormal(),
        #                                      dtype=tf.float32)

    
    def extract_feature(self, x, training=True):
        '''
            x: (b or seq_len, 12, 128, 256).
        '''
        x = tf.transpose(x, perm=[0,2,3,1]) # (b, 128, 256, 12).
        features = self.backbone(x, training=training) # (b, 1024).
        features = self.plan_head(features, training=training) # (b, hid_sz).
        return features # (b, hid_sz).



    def temporal_attention(self, x, training=True):
        '''
            x: (b, seq_len, 768).
        '''                                                    
        shapes = tf.shape(x)
        cls_tokens = tf.broadcast_to(self.cls_token,(shapes[0], 1 ,shapes[2]))
        x = tf.concat((cls_tokens, x), axis=1) # (b, 1+T, 768).

        x += self.pos_embedding

        x = self.transformerEncoder(x, training=training) # (b, 1+T, 768).
 
        x = x[:, 0] # (b, 768).
 
        return x # (b, 768).        



 
    def call(self, x, features_past, training=True):
        '''
            x: (b, 12, 128, 256).
            features_past: (b, seq_len-1, 768).
        '''

        feature_cur = self.extract_feature(x)[:, None, :] # (b, 1, hid_sz).
        
        # print(f'feature_cur.shape: {feature_cur.shape}, flatten: {tf.keras.layers.Flatten()(feature_cur).shape}')

        # x = tf.concat([feature_cur, features_past], axis=1) # (b, seq_len, 768).

        # x = self.temporal_attention(x, training=training) # (b, 768).

        x = self.plan_head_tip(feature_cur, training=training) # (b, 2*num_pts+1).

        # x = tf.math.tanh(x) 

        # # path = tf.math.sinh(x[:, :self.num_pts]) # (b, num_pts).

        # path = x[:, :self.num_pts] # (b, num_pts).
        # std = x[:, self.num_pts:2*self.num_pts] # (b, num_pts).
        # valid_len = x[:, 2*self.num_pts:] # (b, 1). 
 
        # path = tanh_rescale(path, -50.0, 50.0) # (b, num_pts).
        # valid_len = tanh_rescale(valid_len, 0.0, 192.0) # (b, 1). 

        # x = tf.concat([path, std, valid_len], axis=1) # (b, 2*num_pts+1).

        return x # (b, 2*num_pts+1).

 




class MultipleTrajectoryPredictionLoss(tf.keras.Model):

    def __init__(self):
        
        super().__init__()
 
        self.num_pts = 192 # 192.
               
        self.reg_loss = tf.keras.losses.Huber(delta=1.0, reduction='none')

 
    def call(self, traj_pred, traj_true, training=True):
        ''' 
            traj_pred: (b, 2*num_pts+1), 
            traj_true: (b, 2*num_pts+1).
        '''        

        path_true = traj_true[:, :self.num_pts] # (b, num_pts).
        std_true = traj_true[:, self.num_pts:2*self.num_pts] # (b, num_pts).
        valid_len_true = traj_true[:, 2*self.num_pts] # (b,). 

        # b = path_true.shape[0]
        # L = tf.cast(valid_len_true, dtype=tf.int32)[:, None] # (b, 1).
        # s = tf.convert_to_tensor([[i for i in range(1, self.num_pts+1)]]*b) # (b, num_pts).
        # mask = tf.cast(s < L, tf.float32) # (b, num_pts).
        

        path_pred = traj_pred[:, :self.num_pts] # (b, num_pts).
        std_pred = traj_pred[:, self.num_pts:2*self.num_pts] # (b, num_pts).
        valid_len_pred = traj_pred[:, 2*self.num_pts] # (b,).
        
        valid_len_loss = self.reg_loss(valid_len_true, valid_len_pred) # (,).
        # valid_len_loss = tf.math.reduce_mean((valid_len_true - valid_len_pred)**2) # (,).
        
        
        path_loss = self.reg_loss(path_true, path_pred) # (b,). 
        # path_loss = self.reg_loss(path_true*mask, path_pred*mask) # (b,). 
        # path_loss = tf.math.reduce_mean(((path_true-path_pred)*mask)**2, axis=1) # (b,). 
        path_loss = tf.math.reduce_mean(path_loss, axis=0) # (,).


        std_loss = self.reg_loss(std_true, std_pred) # (b,). 
        # std_loss = self.reg_loss(std_true*mask, std_pred*mask) # (b,). 
        # std_loss = tf.math.reduce_mean(((std_true-std_pred)*mask)**2, axis=1) # (b,). 
        std_loss = tf.math.reduce_mean(std_loss, axis=0) # (,).
 
        return path_loss, valid_len_loss, std_loss # (,).
 
  