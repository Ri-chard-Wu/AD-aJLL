

import tensorflow as tf
from EfficientNet import EfficientNetB2
from FastVit import FastViT
from TransformerEncoder import TransformerEncoder
 
from parameter import transformerEncoder_args as enc_args
from parameter import AttrDict

class SequencePlanningNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
      
        num_pts = 192
        self.num_pts = num_pts     
        
        '''EfficientNetB2
            in shape: (b, 12, 128, 256) .
            out shape: (b, 4, 8, 1408).
            ch format: last.
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
  
        for i, layer in enumerate(self.backbone.layers):
            layer.trainable = False
            print(f'[{i}] {layer.name}: layer.trainable = False')
     
        self.plan_head = tf.keras.Sequential([ 
            tf.keras.layers.Dense(enc_args.hidden_size), # (b, hid_sz).
            tf.keras.layers.ELU(),
        ]) # (b, hid_sz).


        self.plan_head_tip = tf.keras.Sequential([
            # tf.keras.layers.Flatten(), 
            # tf.keras.layers.ELU(),  # (b, 768).
            tf.keras.layers.Dense(256, activation='relu'),  # (b, 256).
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'), 
            tf.keras.layers.Dense(num_pts * 2 + 1)
        ]) 
 
        # self.feature_sequence = deque(maxlen=100) # most recent 100 frames as input to transformer encoder.
 
        self.transformerEncoder = TransformerEncoder()
        
        self.cls_token = self.add_weight("cls",
                                         shape=[1,
                                                1,
                                                enc_args.hidden_size],
                                         initializer=tf.keras.initializers.RandomNormal(),
                                         dtype=tf.float32)

        self.pos_embedding = self.add_weight("Transformer/posembed_input/pos_embedding",
                                             shape=[1, enc_args.seq_len + 1, enc_args.hidden_size],
                                             initializer=tf.keras.initializers.RandomNormal(),
                                             dtype=tf.float32)

    
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

        feature_cur = self.extract_feature(x)[:, None, :] # (b, 1, 768).
 
        x = tf.concat([feature_cur, features_past], axis=1) # (b, seq_len, 768).

        x = self.temporal_attention(x, training=training) # (b, 768).

        x = self.plan_head_tip(x, training=training) # (b, 2*num_pts+1).


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

        # path_true = traj_true[:, :self.num_pts] # (b, num_pts).
        # valid_len_true = tf.clip_by_value(traj_true[:, 2*self.num_pts], 5, self.num_pts-1) # (b,).
        
        # L = tf.cast(valid_len_true, dtype=tf.int32) # (b,).

        # path_pred = traj_pred[:, :self.num_pts] # (b, num_pts).
        # valid_len_pred = tf.clip_by_value(traj_pred[:, 2*self.num_pts], 5, self.num_pts-1) # (b,).
        
        # valid_len_loss = self.reg_loss(valid_len_true, valid_len_pred) # (,).
 
        # reg_loss = self.reg_loss(path_true, path_pred) # (b,). 
        # reg_loss = tf.math.reduce_mean(reg_loss, axis=0) # (,).
        
        # return reg_loss + valid_len_loss # (,), (,).

        reg_loss = self.reg_loss(traj_true, traj_pred) # (b,). 
        reg_loss = tf.math.reduce_mean(reg_loss, axis=0) # (,).
        return reg_loss
 
    # def call(self, pred_traj, gt, training=True):
    #     '''
    #         pred_cls_prob: (b, 5), 
    #         pred_trajectory: (b, 5, 2*num_pts+1), 
    #         gt: (b, 2*num_pts+1).
    #     '''        

    #     path_gt = gt[:, :self.num_pts] # (b, num_pts).
    #     valid_len_gt = tf.clip_by_value(gt[:, 2*self.num_pts], 5, self.num_pts-1) # (b,).
    #     L = tf.cast(valid_len_gt, dtype=tf.int32) # (b,).

    #     path_pred = pred_traj[:, :, :self.num_pts] # (b, 5, num_pts).
    #     valid_len_pred = tf.clip_by_value(pred_traj[:, :, 2*self.num_pts], 5, self.num_pts-1) # (b, 5).

        

    #     b = pred_cls_prob.shape[0]
    #     M = self.M


    #     nograd_path_pred = tf.stop_gradient(path_pred)
    #     nograd_path_gt = tf.stop_gradient(path_gt)
    #     nograd_L = tf.stop_gradient(L)
 
    #     sel = tf.stack([tf.repeat(tf.range(b), repeats=M, axis=0),
    #                     tf.convert_to_tensor(list(range(M))*b, dtype=tf.int32), 
    #                     tf.repeat(nograd_L, repeats=M, axis=0)], 
    #                 axis=1)   
    #     pred_end_positions = tf.reshape(tf.gather_nd(nograd_path_pred, sel), (b, M)) # (b, 5).
 
    #     sel = tf.stack([tf.range(b), nograd_L], axis=1)  
    #     gt_end_positions = tf.gather_nd(nograd_path_gt, sel)[:, None] # (b, 1).
 
    #     distances = (pred_end_positions - gt_end_positions) ** 2 # (b, 5).
 
    #     gt_cls = tf.argmin(distances, axis=1) # (b,).
    #     gt_cls = tf.cast(gt_cls, dtype=tf.int32)



    #     # pred_traj = pred_traj[torch.tensor(range(len(gt_cls)),\
    #     #                          device=gt_cls.device), index, ...]  # (b, num_pts, 3).
    #     sel = tf.stack([tf.range(b), gt_cls], axis=1) # (b, 2).
    #     path_pred = tf.gather_nd(path_pred, sel) # (b, num_pts).

    #     valid_len_pred = tf.gather_nd(valid_len_pred, sel) # (b,).
    #     valid_len_loss = self.reg_loss(valid_len_gt, valid_len_pred) # (,).

    #     gt_cls_onehot = tf.one_hot(gt_cls, M) # (b, 5).
    #     cls_loss = self.cls_loss(gt_cls_onehot, pred_cls_prob) # (b,).
    #     reg_loss = self.reg_loss(path_gt, path_pred) # (b,).
 

    #     cls_loss = tf.math.reduce_mean(cls_loss, axis=0) # (,).
    #     reg_loss = tf.math.reduce_mean(reg_loss, axis=0) # (,).
    #     # valid_len_loss = tf.math.reduce_mean(valid_len_loss, axis=0) # (,).

    #     return cls_loss, reg_loss, valid_len_loss # (,), (,).

