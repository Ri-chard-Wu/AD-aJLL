import os
import sys
import time
import random
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from torch.utils.data.distributed import DistributedSampler

 
from torch.utils.tensorboard import SummaryWriter

from data import Comma2k19SequenceDataset
from model import MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric, get_val_metric_keys

from utils import visualize
import glob

# # Training on a bare-metal machine with a single GPU
# PORT=23333 SLURM_PROCID=0 SLURM_NTASKS=1 python main.py

def get_hyperparameters(parser: ArgumentParser):
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_per_n_step', type=int, default=20)
    parser.add_argument('--val_per_n_epoch', type=int, default=1)

    parser.add_argument('--resume', type=str, default='')

    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--num_pts', type=int, default=33)
    parser.add_argument('--mtp_alpha', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--sync_bn', type=bool, default=True)
    parser.add_argument('--tqdm', type=bool, default=False)
    parser.add_argument('--optimize_per_n_step', type=int, default=8)

    try:
        exp_name = os.environ["SLURM_JOB_ID"]
    except KeyError:
        exp_name = str(time.time())
    parser.add_argument('--exp_name', type=str, default=exp_name)

    return parser

 




def get_dataloader(batch_size, pin_memory=False, num_workers=0):


    paths = glob.glob("./data/comma2k19-LD/scb*")
    split = int(len(paths)*0.95)

    train = Comma2k19SequenceDataset(paths[:split], 'train')

    # train[1]
    # exit()
    
    val = Comma2k19SequenceDataset(paths[split:], 'demo', return_origin=True)

    
    dist_sampler_params = dict(shuffle=True, drop_last=True)
        
    loader_args = dict(num_workers=num_workers, 
                        # persistent_workers=True, 
                        # prefetch_factor=2, 
                        pin_memory=pin_memory,
                        shuffle=True, 
                        drop_last=True)

    print(f'>>>> batch_size: {batch_size}')
    train_loader = DataLoader(train, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val, batch_size=1, **loader_args)

    return train_loader, val_loader


 

class SequenceBaselineV1(nn.Module):
    def __init__(self, M, num_pts, mtp_alpha, lr, optimizer, optimize_per_n_step=40) -> None:
        super().__init__()
        self.M = M # 5.
        self.num_pts = num_pts # 33.
        self.mtp_alpha = mtp_alpha
        self.lr = lr
        self.optimizer = optimizer # 'sgd'.

        self.net = SequencePlanningNetwork(M, num_pts)

        self.optimize_per_n_step = optimize_per_n_step  # for the gru module

    @staticmethod
    def configure_optimizers(args, model):
        if args.optimizer == 'sgd': # yes.
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, )
        else:
            raise NotImplementedError
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9) # lr = lr * 0.9 every 20 epochs.

        return optimizer, lr_scheduler

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros((2, x.size(0), 512)).to(self.device)
        return self.net(x, hidden) # (b, M), (b, M, num_pts, 3), .






def main(args):

    writer = SummaryWriter()

    train_dataloader, val_dataloader = get_dataloader(args.batch_size, False, args.n_workers)
    
    model = SequenceBaselineV1(args.M, args.num_pts, args.mtp_alpha, 
                args.lr, args.optimizer, args.optimize_per_n_step) 
    
    # use_sync_bn = args.sync_bn 
    # if use_sync_bn:
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    optimizer, lr_scheduler = model.configure_optimizers(args, model)
   
    if args.resume:
        print('Loading weights from', args.resume)
        model.load_state_dict(torch.load(args.resume), strict=True)

 
    loss = MultipleTrajectoryPredictionLoss(args.mtp_alpha, args.M, args.num_pts, distance_type='angle')

    num_steps = 0
    disable_tqdm = True







    for epoch in tqdm(range(args.epochs), disable=disable_tqdm, position=0):

    #     for batch_idx, data in enumerate(tqdm(train_dataloader, leave=False, disable=disable_tqdm, position=1)):
            
            

    #         # seq_inputs: (b, N+1, 6, H, W), 2 rgb, not yuv.
    #         seq_inputs, seq_labels = data['seq_input_img'].cuda(),\
    #                              data['seq_future_poses'].cuda() # (b, N+1, 6, h, w), (b, N, num_pts, 3).
            
    #         bs = seq_labels.size(0) 
    #         seq_length = seq_labels.size(1)  
            
    #         hidden = torch.zeros((2, bs, 256)).cuda()
    #         total_loss = 0

    #         total_losses = []

    #         for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm, position=2):
                
    #             num_steps += 1

    #             inputs, labels = seq_inputs[:, t, :, :, :], \
    #                     seq_labels[:, t, :, :] # (b, 6, h, w), (b, num_pts, 3).
                
    #             # inputs: (b, 6, H, W), 2 rgb, not yuv.
    #             pred_cls, pred_trajectory, hidden = model(inputs, hidden) # (b, M), (b, M, num_pts, 3), .
                
    #             cls_loss, reg_loss = loss(pred_cls, pred_trajectory, labels) # (,)?, (3,).

    #             total_loss += (cls_loss + args.mtp_alpha * reg_loss.mean()) / args.optimize_per_n_step
            
    #             if (num_steps + 1) % args.log_per_n_step == 0:
    #                 # TODO: add a customized log function
    #                 writer.add_scalar('train/epoch', epoch, num_steps)
    #                 writer.add_scalar('loss/cls', cls_loss, num_steps)
    #                 writer.add_scalar('loss/reg', reg_loss.mean(), num_steps)
    #                 writer.add_scalar('loss/reg_x', reg_loss[0], num_steps)
    #                 writer.add_scalar('loss/reg_y', reg_loss[1], num_steps)
    #                 writer.add_scalar('loss/reg_z', reg_loss[2], num_steps)
    #                 writer.add_scalar('param/lr', optimizer.param_groups[0]['lr'], num_steps)
                    
    #                 # print(f"[{num_steps}] cls_loss: {cls_loss}, reg: {reg_loss.mean()}, lr: {optimizer.param_groups[0]['lr']}")



    #             if (t + 1) % args.optimize_per_n_step == 0:
    #                 hidden = hidden.clone().detach()
    #                 optimizer.zero_grad()
    #                 total_loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
    #                 optimizer.step()
    #                 writer.add_scalar('loss/total', total_loss, num_steps)     

    #                 total_losses.append(total_loss.detach().cpu().numpy())


    #                 total_loss = 0


    #         print(f'[{num_steps}] total_loss: {np.mean(total_losses)}')

    #         if not isinstance(total_loss, int):
    #             optimizer.zero_grad()
    #             total_loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
    #             optimizer.step()

    #             writer.add_scalar('loss/total', total_loss, num_steps)

      
    #     lr_scheduler.step()



        if (epoch + 1) % args.val_per_n_epoch == 0:
 
            ckpt_path = os.path.join(writer.log_dir, 'epoch_%d.pth' % epoch)
            torch.save(model.state_dict(), ckpt_path)
            print('[Epoch %d] checkpoint saved at %s' % (epoch, ckpt_path))

            model.eval()

            with torch.no_grad():

                saved_metric_epoch = get_val_metric_keys()

                for batch_idx, data in enumerate(tqdm(val_dataloader, 
                                   leave=False, disable=disable_tqdm, position=1)):
                    

                    seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()
                    origin_imgs = data['origin_imgs']

                    bs = seq_labels.size(0)
                    seq_length = seq_labels.size(1)
                    
                    
                    hidden = torch.zeros((2, bs, 256), device=seq_inputs.device)
                    for t in tqdm(range(seq_length), leave=False, disable=True, position=2):
                        
                        print(f'[{batch_idx}] seq_length: {seq_length}, t: {t}')

                        inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]

                        pred_cls, pred_trajectory, hidden = model(inputs, hidden)

                        metrics = get_val_metric(pred_cls, pred_trajectory.view(-1, args.M, args.num_pts, 3), labels)
                        
                        for k, v in metrics.items():
                            saved_metric_epoch[k].append(v.float().mean().item())
                        
                        pred_conf = softmax(pred_cls, dim=-1).cpu().numpy()[0] # (M,).
                        pred_trajectory = pred_trajectory.reshape(args.M, args.num_pts, 3).cpu().numpy() # (M, num_pts, 3).                

                        # print(f'origin_img.dtype: {origin_imgs[0, t, :, :, :].numpy().dtype}, mean: {np.mean(origin_imgs[0, t, :, :, :].numpy())}')

                        # if(t%40==0):
                        #     visualize(origin_imgs[0, t, :, :, :].numpy(), 
                        #         inputs.cpu(), # (1, 6, h, w).
                        #         labels.cpu(), # (1, num_pts, 3).
                        #         pred_trajectory, # (M, num_pts, 3).
                        #         pred_conf, # (M,).
                        #         dir_name='output/val',
                        #         file_name='%04d-%04d.jpg' % (batch_idx, t)
                        #         )

                    del data
                    del origin_imgs


                metric_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')
                counter_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda')
                
                for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                    writer.add_scalar(k, np.mean(saved_metric_epoch[k]), num_steps)
              
            model.train()

 




if __name__ == "__main__":
 
    parser = ArgumentParser()
    parser = get_hyperparameters(parser)
    args = parser.parse_args()

    # setup(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']))
    main(args=args)
