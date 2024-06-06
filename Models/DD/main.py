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

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

if torch.__version__ == 'parrots':
    from pavi import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

from data import PlanningDataset, SequencePlanningDataset, Comma2k19SequenceDataset
from model import MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric, get_val_metric_keys







PATH_DISTANCE = 192
LANE_OFFSET = 1.8
LEAD_X_SCALE = 10   # x_scale in driving.cc
LEAD_Y_SCALE = 10   # y_scale in driving.cc
LEAD_V_SCALE = 1

TIME_DISTANCE = 100
 
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
 

def get_hyperparameters(parser: ArgumentParser):
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=4)
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
    parser.add_argument('--optimize_per_n_step', type=int, default=40)

    try:
        exp_name = os.environ["SLURM_JOB_ID"]
    except KeyError:
        exp_name = str(time.time())
    parser.add_argument('--exp_name', type=str, default=exp_name)

    return parser


def setup(rank, world_size):

    torch.cuda.set_device(rank)

    dist.init_process_group('nccl', init_method='tcp://localhost:%s' % os.environ['PORT'], rank=rank, world_size=world_size)

    print('[%.2f]' % time.time(), 'DDP Initialized at %s:%s' % ('localhost', os.environ['PORT']), rank, 'of', world_size, flush=True)




# def get_dataloader(rank, world_size, batch_size, pin_memory=False, num_workers=0):
#     train = Comma2k19SequenceDataset('data/comma2k19_train_non_overlap.txt', 'data/comma2k19/','train', use_memcache=False)
#     val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/','demo', use_memcache=False)

#     if torch.__version__ == 'parrots':
#         dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True)
#     else:
#         dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        
#     train_sampler = DistributedSampler(train, **dist_sampler_params)
#     val_sampler = DistributedSampler(val, **dist_sampler_params)

#     loader_args = dict(num_workers=num_workers, persistent_workers=True if num_workers > 0 else False, prefetch_factor=2, pin_memory=pin_memory)
#     train_loader = DataLoader(train, batch_size, sampler=train_sampler, **loader_args)
#     val_loader = DataLoader(val, batch_size=1, sampler=val_sampler, **loader_args)

#     return train_loader, val_loader




def get_train_dataloader(pkl_files):
  
    hevc_files = [pkl_file.replace('data.pkl', 'fcamera.hevc') for pkl_file in pkl_files]    

    H = para.horizon 
    B = para.batch_size
    n_B = len(pkl_files) // B 
         
    reps = 1

    while(1): 

        gfidx = 0

        for bidx in range(n_B):

            if((bidx+1) * B > len(pkl_files)): break
 
            for ridx in range(reps): 
    
                X0 = np.zeros((B, H, 12, 128, 256), dtype=np.uint8)            
                X3 = np.zeros((B, H, 512), dtype=np.float32)    
                Y = [np.zeros((B, H, s)) for s in Y_shape]
    
                progress_bar = tqdm(total=B, desc="sampleing training data...")
                
                
                for fidx in range(B):
                    
                    progress_bar.update(1)


                    while(1):
                        pkl_file = pkl_files[gfidx]
                        hevc_file = hevc_files[gfidx]                    
                        frames = read_frames(hevc_file) 

                        gfidx += 1
                        if(gfidx >= len(pkl_files)): gfidx = 0

                        if(len(frames)-H > 0):  
                            break
                        else:
                            print(f'[Warning] len(frames)-H <= 0): {hevc_file}')                        
                            continue


                    
                    
                    t0 = np.random.choice(np.arange(len(frames)-H), size=1)[0]
                    frames = frames[t0:t0+H+1]
    
                    with open(pkl_file, 'rb') as f:      

                        data = pickle.load(f)   
                        assert data['Xin3'].shape[1] == 512
                           
                        for i in range(12):
                            assert data['Y'][i].shape[1] == Y_shape[i]
    
                    for t in range(H):   
            
                        X0[fidx, t] = np.vstack((RGB_to_YUV(frames[t]), RGB_to_YUV(frames[t+1])))
                        X3[fidx, t] = data['Xin3'][t0+t]
                        for j in range(12):
                            Y[j][fidx, t] = data['Y'][j][t0+t]


                    # fidx += 1


                yield X0, Y[0] # (b, T, 12, 128, 256), (b, T, 2*num_pts+1).

                del X0, Y


 

class SequenceBaselineV1(nn.Module):
    def __init__(self, M, num_pts, mtp_alpha, lr, optimizer, optimize_per_n_step=40) -> None:
        super().__init__()
        self.M = M # 5.
        self.num_pts = num_pts # 33.
        self.mtp_alpha = mtp_alpha
        self.lr = lr
        self.optimizer = optimizer # 'sgd'.

        self.net = SequencePlanningNetwork(args.M, args.num_pts)

        self.optimize_per_n_step = optimize_per_n_step  # for the gru module

    # @staticmethod
    # def configure_optimizers(args, model):
      
    #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9) # lr = lr * 0.9 every 20 epochs.

    #     return optimizer, lr_scheduler

    def forward(self, x, hidden=None):
 
        return self.net(x, hidden) # (b, M), (b, M, num_pts, 3), .



def main(rank, world_size, args):
 
  
    # model = SequenceBaselineV1(args.M, args.num_pts, args.mtp_alpha, 
    #             args.lr, args.optimizer, args.optimize_per_n_step) 
    
    model = SequencePlanningNetwork(args.M, args.num_pts)

    # optimizer, lr_scheduler = model.configure_optimizers(args, model)
 
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr,
                    weight_decay=0.01, ema_momentum=0.9, clipvalue=1.0)        


    # if args.resume and rank == 0:
    #     print('Loading weights from', args.resume)
    #     model.load_state_dict(torch.load(args.resume), strict=True)


    loss = MultipleTrajectoryPredictionLoss(args.mtp_alpha, args.M, args.num_pts, distance_type='angle')
 

    train_dataloader = get_train_dataloader(train_pkl)


    for epoch in range(args.epochs):
 
        for epid, data in enumerate(train_dataloader):  
    
            seq_inputs, seq_labels = data # (b, T, 12, 128, 256), (b, T, 2*num_pts+1). 

            bs = seq_labels.shape[0]
            seq_length = seq_labels.shape[1]
            
            hidden = (tf.zeros((bs, 512)), tf.zeros((bs, 512)))
 
            H1 = args.optimize_per_n_step
            n1 = seq_length//H1
            for t1 in range(n1):
                
                loss = 0

                with tf.GradientTape() as tape:

                    for t in range(t*n1, (t+1)*n1):

                        num_steps += 1

                        inputs, labels = seq_inputs[:, t, :, :, :], \
                                seq_labels[:, t, :] # (b, 12, 128, 256), (b, 2*num_pts+1).
                                        
                        pred_cls, pred_trajectory, hidden = model(inputs, hidden) # (b, M), (b, M, num_pts, 3), .
                        
                        cls_loss, reg_loss = loss(pred_cls, pred_trajectory, labels) # (,), (,).

                        loss += (cls_loss + args.mtp_alpha * reg_loss) / H1 # "/H1" may cause error.
                
                
                grad = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grad, model.trainable_variables))

                
                hidden = [tf.convert_to_tensor(hidden[0].numpy(), dtype=tf.float32),
                          tf.convert_to_tensor(hidden[1].numpy(), dtype=tf.float32)]

        # lr_scheduler.step()




    ##################

    # for epoch in range(args.epochs):

    #     train_dataloader.sampler.set_epoch(epoch)
        
    #     for batch_idx, data in enumerate(train_dataloader):

    #         # seq_inputs: (b, N+1, 6, H, W), 2 rgb, not yuv.
    #         seq_inputs, seq_labels = data['seq_input_img'].cuda(),\
    #                              data['seq_future_poses'].cuda() # (b, N+1, 6, h, w), (b, N, num_pts, 3).
            
    #         bs = seq_labels.shape[0]
    #         seq_length = seq_labels.shape[1]
            
    #         hidden = (tf.zeros((bs, 512)), tf.zeros((bs, 512)))

    #         total_loss = 0

    #         for t in range(seq_length):
                
    #             num_steps += 1

    #             inputs, labels = seq_inputs[:, t, :, :, :], \
    #                     seq_labels[:, t, :, :] # (b, 6, h, w), (b, num_pts, 3).
                
    #             # inputs: (b, 6, H, W), 2 rgb, not yuv.
    #             pred_cls, pred_trajectory, hidden = model(inputs, hidden) # (b, M), (b, M, num_pts, 3), .
                
    #             cls_loss, reg_loss = loss(pred_cls, pred_trajectory, labels) # (,), (,).

    #             total_loss += (cls_loss + args.mtp_alpha * reg_loss) / model.module.optimize_per_n_step
            
               
    #             if (t + 1) % model.module.optimize_per_n_step == 0:
    #                 hidden = hidden.clone().detach()
    #                 optimizer.zero_grad()
    #                 total_loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
    #                 optimizer.step() 
    #                 total_loss = 0
 

    #         if not isinstance(total_loss, int):
    #             optimizer.zero_grad()
    #             total_loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
    #             optimizer.step()
 


    #     lr_scheduler.step()

    #     if (epoch + 1) % args.val_per_n_epoch == 0:

    #         if rank == 0:
    #             # save model
    #             ckpt_path = os.path.join(writer.log_dir, 'epoch_%d.pth' % epoch)
    #             torch.save(model.module.state_dict(), ckpt_path)
    #             print('[Epoch %d] checkpoint saved at %s' % (epoch, ckpt_path))

    #         model.eval()

    #         with torch.no_grad():

    #             saved_metric_epoch = get_val_metric_keys()

    #             for batch_idx, data in enumerate(val_dataloader):
                    
    #                 seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()

    #                 bs = seq_labels.size(0)
    #                 seq_length = seq_labels.size(1)
                    
    #                 hidden = torch.zeros((2, bs, 512), device=seq_inputs.device)
    #                 for t in tqdm(range(seq_length), leave=False, disable=True, position=2):

    #                     inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]

    #                     pred_cls, pred_trajectory, hidden = model(inputs, hidden)

    #                     metrics = get_val_metric(pred_cls, pred_trajectory.view(-1,
    #                                                     args.M, args.num_pts, 3), labels)
                        
    #                     for k, v in metrics.items():
    #                         saved_metric_epoch[k].append(v.float().mean().item())
                
             
    #             metric_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')
    #             counter_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda')

    #             # From Python 3.6 onwards, the standard dict type maintains insertion order by default.
    #             # But, programmers should not rely on it.
    #             for i, k in enumerate(sorted(saved_metric_epoch.keys())):
    #                 metric_single[i] = np.mean(saved_metric_epoch[k])
    #                 counter_single[i] = len(saved_metric_epoch[k])

    #             metric_gather = [torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')[None] for _ in range(world_size)]
    #             counter_gather = [torch.zeros((len(saved_metric_epoch), ),
    #                             dtype=torch.int32, device='cuda')[None] for _ in range(world_size)]
               
    #             if rank == 0:
    #                 metric_gather = torch.cat(metric_gather, dim=0)  # [world_size, num_metric_keys]
    #                 counter_gather = torch.cat(counter_gather, dim=0)  # [world_size, num_metric_keys]
    #                 metric_gather_weighted_mean = (metric_gather * counter_gather).sum(0) / counter_gather.sum(0)
    #                 for i, k in enumerate(sorted(saved_metric_epoch.keys())):
    #                     writer.add_scalar(k, metric_gather_weighted_mean[i], num_steps)
               

    #         model.train()

 



if __name__ == "__main__":
    print('[%.2f]' % time.time(), 'starting job...', os.environ['SLURM_PROCID'], 'of', os.environ['SLURM_NTASKS'], flush=True)

    parser = ArgumentParser()
    parser = get_hyperparameters(parser)
    args = parser.parse_args()

    setup(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']))
    main(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']), args=args)
