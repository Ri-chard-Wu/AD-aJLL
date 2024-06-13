import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet





class SequencePlanningNetwork(nn.Module):
    def __init__(self, M, num_pts):
        super().__init__()
      
        self.M = M # 5.
        self.num_pts = num_pts # 33.    
        self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=6)

        self.plan_head = nn.Sequential(
            # 6, 450, 800 -> 1408, 14, 25
            # nn.AdaptiveMaxPool2d((4, 8)),  # 1408, 4, 8
            nn.BatchNorm2d(1408),
            nn.Conv2d(1408, 32, 1),  # 32, 4, 8
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ELU(),
        )

        self.gru = nn.GRU(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)  # 1024 out
        
        self.plan_head_tip = nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(4096),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(512, M * (num_pts * 3 + 1))  # +1 for cls
        )


    def forward(self, x, hidden):

        features = self.backbone.extract_features(x)

        raw_preds = self.plan_head(features)

        raw_preds, hidden = self.gru(raw_preds[:, None, :], hidden)  # N, L, H_in for batch_first=True
        
        raw_preds = self.plan_head_tip(raw_preds)

        pred_cls = raw_preds[:, :self.M] # (b, 5).
        pred_trajectory = raw_preds[:, self.M:].reshape(-1, self.M, self.num_pts, 3) # (b, 5, 33, 3).

        pred_xs = pred_trajectory[:, :, :, 0:1].exp() # (b, 5, 33, 1).
        pred_ys = pred_trajectory[:, :, :, 1:2].sinh() # (b, 5, 33, 1).
        pred_zs = pred_trajectory[:, :, :, 2:3] # (b, 5, 33, 1).

        return pred_cls, torch.cat((pred_xs, pred_ys, pred_zs), dim=3), hidden # (b, M), (b, M, num_pts, 3), .




class AbsoluteRelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        error = (pred - target) / (target + self.epsilon)
        return torch.abs(error)



class SigmoidAbsoluteRelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        error = (pred - target) / (target + self.epsilon)
        return torch.sigmoid(torch.abs(error))



class MultipleTrajectoryPredictionLoss(nn.Module):

    def __init__(self, alpha, M, num_pts, distance_type='angle'):
        super().__init__()
        self.alpha = alpha  # TODO: currently no use
        self.M = M
        self.num_pts = num_pts
        
        self.distance_type = distance_type

        if self.distance_type == 'angle':
            self.distance_func = nn.CosineSimilarity(dim=2)
        else:
            raise NotImplementedError
        
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        # self.reg_loss = SigmoidAbsoluteRelativeErrorLoss()
        # self.reg_loss = AbsoluteRelativeErrorLoss()


    def forward(self, pred_cls, pred_trajectory, gt):
        """
        pred_cls: [B, M]
        pred_trajectory: [B, M * num_pts * 3]
        gt: [B, num_pts, 3]
        """

        '''
            pred_cls: (b, M), 
            pred_trajectory: (b, M, num_pts, 3), 
            gt: (b, num_pts, 3).
        '''        

        assert len(pred_cls) == len(pred_trajectory) == len(gt)

        pred_trajectory = pred_trajectory.reshape(-1, self.M, self.num_pts, 3) # (b, M, num_pts, 3).

        with torch.no_grad():
            # step 1: calculate distance between gt and each prediction
            pred_end_positions = pred_trajectory[:, :, self.num_pts-1, :]  # (b, M, 3).
            gt_end_positions = gt[:, self.num_pts-1:, :].expand(-1, self.M, -1)  # (b, M, 3).
            
            distances = 1 - self.distance_func(pred_end_positions, gt_end_positions)  # (b, M).
            index = distances.argmin(dim=1)  # (b,).


        gt_cls = index  # (b,).
        pred_trajectory = pred_trajectory[torch.tensor(range(len(gt_cls)),\
                                 device=gt_cls.device), index, ...]  # (b, num_pts, 3).

        cls_loss = self.cls_loss(pred_cls, gt_cls) # (,)?.
        reg_loss = self.reg_loss(pred_trajectory, gt).mean(dim=(0, 1)) # (3,).

        return cls_loss, reg_loss # (,)?, (3,).

 