import torch
import torch.nn as nn
import torch.nn.functional as F

# from efficientnet_pytorch import EfficientNet
from EfficientNet import EfficientNetB2



# class SequencePlanningNetwork(nn.Module):
#     def __init__(self, M, num_pts):
#         super().__init__()
      
#         self.M = M # 5.
#         self.num_pts = num_pts # 33.    


#         self.backbone = EfficientNetB2()

#         self.plan_head = nn.Sequential(
#             # 6, 450, 800 -> 1408, 14, 25
#             # nn.AdaptiveMaxPool2d((4, 8)),  # 1408, 4, 8
#             nn.BatchNorm2d(1408),
#             nn.Conv2d(in_channels=1408, out_channels=32, kernel_size=1),  # (b, 32, 4, 8).
#             nn.BatchNorm2d(32),
#             nn.Flatten(), # (b, 1024).
#             nn.ELU(),
#         ) # (b, 1024).

#         self.gru = nn.GRU(input_size=1024, hidden_size=512, bidirectional=True, batch_first=True)  # 1024 out
#         self.plan_head_tip = nn.Sequential(
#             nn.Flatten(),
#             # nn.BatchNorm1d(1024),
#             nn.ELU(),
#             nn.Linear(1024, 4096),
#             # nn.BatchNorm1d(4096),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(4096, M * (num_pts * 3 + 1))  # +1 for cls
#         )


#     def forward(self, x, hidden):
#         '''
#             x: (b, 6, H, W).
#         '''

#         features = self.backbone.extract_features(x) # (b, 1408, 4, 8).

#         raw_preds = self.plan_head(features) # (b, 1024).

#         raw_preds, hidden = self.gru(raw_preds[:, None, :], hidden)  # N, L, H_in for batch_first=True
#         raw_preds = self.plan_head_tip(raw_preds) # (b, M+M*num_pts*3).

#         pred_cls = raw_preds[:, :self.M] # (b, 5).
#         pred_trajectory = raw_preds[:, self.M:].reshape(-1, self.M, self.num_pts, 3) # (b, 5, 33, 3).

#         pred_xs = pred_trajectory[:, :, :, 0:1].exp() # (b, 5, 33, 1).
#         pred_ys = pred_trajectory[:, :, :, 1:2].sinh() # (b, 5, 33, 1).
#         pred_zs = pred_trajectory[:, :, :, 2:3] # (b, 5, 33, 1).

#         return pred_cls, torch.cat((pred_xs, pred_ys, pred_zs), dim=3), hidden # (b, M), (b, M, num_pts, 3), .

 




class SequencePlanningNetwork(tf.keras.Model):
    def __init__(self, M, num_pts):
        super().__init__()
      
        self.M = M # 5.
        
        num_pts = 192
        self.num_pts = num_pts     
        
            

        '''EfficientNetB2
            in shape: (b, 128, 256, 3) .
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
  
        self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_state=True, return_sequences=True))
 
        self.plan_head_tip = tf.keras.Sequential([
            tf.keras.layers.Flatten(),  # (b, 1024).
            tf.keras.layers.ELU(),
            tf.keras.layers.Dense(4096),  # (b, 4096).
            tf.keras.layers.ReLU(), 
            tf.keras.layers.Dense(M + M * (num_pts * 2 + 1)) 
        ]) 
 

    def call(self, x, hidden, training=True):
        '''
            x: (b, 6, H, W).
        '''

        features = self.backbone(x, training=training) # (b, 1408, 4, 8).

        raw_preds = self.plan_head(features, training=training) # (b, 1024).


        hidden_fw, hidden_bw = hidden # (b, 512), (b, 512).

 
        raw_preds, hidden_fw, hidden_bw = self.gru(raw_preds[:, None, :], initial_state=[hidden_fw, hidden_bw], training=training) # (b, 1, 1024), (b, 512), (b, 512).

        raw_preds = self.plan_head_tip(raw_preds, training=training) # (b, M+M*(2*num_pts+1)).

        pred_cls = tf.nn.softmax(raw_preds[:, :self.M]) # (b, 5).

        pred_traj = tf.reshape(raw_preds[:, self.M:],\
                             (-1, self.M, 2*self.num_pts+1)) # (b, 5, 2*num_pts+1).

        # path = pred_traj[:, :, :self.num_pts] # (b, 5, num_pts).
        # path_std = tf.math.softplus(pred_traj[:, :, self.num_pts:2*self.num_pts]) # (b, 5, num_pts).
        # valid_len = tf.clip_by_value(pred_traj[:, :, 2*self.num_pts:], 5, 192) # (b, 5, 1).

        # pred_traj = tf.concat([path, path_std, valid_len], axis=2) # (b, 5, 2*num_pts+1).

        return pred_cls, pred_traj, (hidden_fw, hidden_bw) # (b, 5), (b, 5, 2*num_pts+1), .

 






class MultipleTrajectoryPredictionLoss(tf.keras.Model):

    def __init__(self, alpha, M, num_pts, distance_type='angle'):
        
        super().__init__()

        self.alpha = alpha  # TODO: currently no use
        self.M = M
        self.num_pts = num_pts
        
        self.distance_type = distance_type

        if self.distance_type == 'angle': # yes.
            self.distance_func = nn.CosineSimilarity(dim=2)
        else:
            raise NotImplementedError
        
        self.cls_loss = tf.keras.losses.CategoricalCrossentropy(reduction='none')        
        # self.reg_loss = nn.SmoothL1Loss(reduction='none')
        self.reg_loss = tf.keras.losses.Huber(delta=1.0)

      
 
    def call(self, pred_cls_prob, pred_traj, gt, training=True):
        '''
            pred_cls_prob: (b, 5), 
            pred_trajectory: (b, 5, 2*num_pts+1), 
            gt: (b, 2*num_pts+1).
        '''        

        # assert len(pred_cls) == len(pred_trajectory) == len(gt)

        # pred_trajectory = pred_trajectory.reshape(-1, self.M, self.num_pts, 3) # (b, M, num_pts, 3).

        path_gt = gt[:, :self.num_pts] # (b, num_pts).
        valid_len_gt = tf.clip_by_value(gt[:, 2*self.num_pts], 5, 192-1) # (b,).
        L = tf.cast(valid_len_gt, dtype=tf.int32) # (b,).

        path_pred = pred_traj[:, :, :self.num_pts] # (b, 5, num_pts).
        valid_len_pred = tf.clip_by_value(pred_traj[:, :, 2*self.num_pts], 5, 192) # (b, 5).

        b = pred_cls_prob.shape[0]
        M = self.M


        with torch.no_grad():
            
            # pred_end_positions = pred_traj[:, :, self.num_pts-1, :]  # (b, M, 3).
            sel = tf.stack([tf.repeat(tf.range(b), repeats=M, axis=0),
                            tf.convert_to_tensor(list(range(M))*b, dtype=tf.int32), 
                            tf.repeat(L, repeats=M, axis=0)], 
                        axis=1)   
            pred_end_positions = tf.reshape(tf.gather_nd(path_pred, sel), (b, M)) # (b, 5).

            # gt_end_positions = gt[:, self.num_pts-1:, :].expand(-1, self.M, -1)  # (b, M, 3).
            sel = tf.stack([tf.range(b), L], axis=1)  
            gt_end_positions = tf.gather_nd(path_gt, sel)[:, None] # (b, 1).

            
            # distances = 1 - self.distance_func(pred_end_positions, gt_end_positions)  # (b, M).

            distances = (pred_end_positions - gt_end_positions) ** 2 # (b, 5).

            # index = distances.argmin(dim=1)  # (b,).
            gt_cls = tf.argmin(distances, axis=1) # (b,).

 
        # pred_traj = pred_traj[torch.tensor(range(len(gt_cls)),\
        #                          device=gt_cls.device), index, ...]  # (b, num_pts, 3).
        sel = tf.stack([tf.range(b), gt_cls], axis=1) 
        path_pred = tf.gather_nd(path_pred, sel) # (b, num_pts).

        gt_cls_onehot = tf.one_hot(gt_cls, M) # (b, 5).
        cls_loss = self.cls_loss(gt_cls_onehot, pred_cls_prob) # (b,).
        reg_loss = self.reg_loss(path_gt, path_pred) # (b,).
 

        cls_loss = tf.math.reduce_mean(cls_loss, axis=0) # (,).
        reg_loss = tf.math.reduce_mean(reg_loss, axis=0) # (,).

        return cls_loss, reg_loss # (,), (,).








# class MultipleTrajectoryPredictionLoss(nn.Module):

#     def __init__(self, alpha, M, num_pts, distance_type='angle'):
        
#         super().__init__()

#         self.alpha = alpha  # TODO: currently no use
#         self.M = M
#         self.num_pts = num_pts
        
#         self.distance_type = distance_type

#         if self.distance_type == 'angle':
#             self.distance_func = nn.CosineSimilarity(dim=2)
#         else:
#             raise NotImplementedError
        
#         self.cls_loss = nn.CrossEntropyLoss()
#         self.reg_loss = nn.SmoothL1Loss(reduction='none')
      

#     def forward(self, pred_cls, pred_trajectory, gt):
#         """
#         pred_cls: [B, M]
#         pred_trajectory: [B, M * num_pts * 3]
#         gt: [B, num_pts, 3]
#         """

#         '''
#             pred_cls: (b, M), 
#             pred_trajectory: (b, M, num_pts, 3), 
#             gt: (b, num_pts, 3).
#         '''        

#         assert len(pred_cls) == len(pred_trajectory) == len(gt)

#         pred_trajectory = pred_trajectory.reshape(-1, self.M, self.num_pts, 3) # (b, M, num_pts, 3).

#         with torch.no_grad():
#             # step 1: calculate distance between gt and each prediction
#             pred_end_positions = pred_trajectory[:, :, self.num_pts-1, :]  # (b, M, 3).
#             gt_end_positions = gt[:, self.num_pts-1:, :].expand(-1, self.M, -1)  # (b, M, 3).
            
#             distances = 1 - self.distance_func(pred_end_positions, gt_end_positions)  # (b, M).
#             index = distances.argmin(dim=1)  # (b,).


#         gt_cls = index  # (b,).
#         pred_trajectory = pred_trajectory[torch.tensor(range(len(gt_cls)),\
#                                  device=gt_cls.device), index, ...]  # (b, num_pts, 3).

#         cls_loss = self.cls_loss(pred_cls, gt_cls) # (,)?.
#         reg_loss = self.reg_loss(pred_trajectory, gt).mean(dim=(0, 1)) # (3,).

#         return cls_loss, reg_loss # (,)?, (3,).




if __name__ == '__main__':
    # model = EfficientNet.from_pretrained('efficientnet-b2', in_channels=6)
    model = PlaningNetwork(M=3, num_pts=20)

    dummy_input = torch.zeros((1, 6, 256, 512))

    # features = model.extract_features(dummy_input)
    features = model(dummy_input)

    pred_cls = torch.rand(16, 5)
    pred_trajectory = torch.rand(16, 5*20*3)
    gt = torch.rand(16, 20, 3)

    loss = MultipleTrajectoryPredictionLoss(1.0, 5, 20)

    loss(pred_cls, pred_trajectory, gt)
