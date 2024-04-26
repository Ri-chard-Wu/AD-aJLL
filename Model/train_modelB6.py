 

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
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric, get_val_metric_keys


# # Training on a bare-metal machine with a single GPU
# PORT=23333 SLURM_PROCID=0 SLURM_NTASKS=1 python main.py

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




def get_dataloader(rank, world_size, batch_size, pin_memory=False, num_workers=0):
    train = Comma2k19SequenceDataset('data/comma2k19_train_non_overlap.txt', 'data/comma2k19/','train', use_memcache=False)
    val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/','demo', use_memcache=False)

    if torch.__version__ == 'parrots':
        dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_sampler = DistributedSampler(train, **dist_sampler_params)
    val_sampler = DistributedSampler(val, **dist_sampler_params)

    loader_args = dict(num_workers=num_workers, persistent_workers=True if num_workers > 0 else False, prefetch_factor=2, pin_memory=pin_memory)
    train_loader = DataLoader(train, batch_size, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val, batch_size=1, sampler=val_sampler, **loader_args)

    return train_loader, val_loader




def cleanup():
    dist.destroy_process_group()



class SequenceBaselineV1(nn.Module):
    def __init__(self, M, num_pts, mtp_alpha, lr, optimizer, optimize_per_n_step=40) -> None:
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.mtp_alpha = mtp_alpha
        self.lr = lr
        self.optimizer = optimizer

        self.net = SequencePlanningNetwork(M, num_pts)

        self.optimize_per_n_step = optimize_per_n_step  # for the gru module

    @staticmethod
    def configure_optimizers(args, model):
        if args.optimizer == 'sgd':
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
        return self.net(x, hidden)






def main(rank, world_size, args):

    if rank == 0:
        writer = SummaryWriter()

    train_dataloader, val_dataloader = get_dataloader(rank, world_size, args.batch_size, False, args.n_workers)
    model = SequenceBaselineV1(args.M, args.num_pts, args.mtp_alpha, args.lr, args.optimizer, args.optimize_per_n_step)
    use_sync_bn = args.sync_bn

    if use_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    optimizer, lr_scheduler = model.configure_optimizers(args, model)
    model: SequenceBaselineV1

    if args.resume and rank == 0:
        print('Loading weights from', args.resume)
        model.load_state_dict(torch.load(args.resume), strict=True)

    dist.barrier()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    loss = MultipleTrajectoryPredictionLoss(args.mtp_alpha, args.M, args.num_pts, distance_type='angle')

    num_steps = 0
    disable_tqdm = (not args.tqdm) or (rank != 0)

    for epoch in tqdm(range(args.epochs), disable=disable_tqdm, position=0):
        train_dataloader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(tqdm(train_dataloader, leave=False, disable=disable_tqdm, position=1)):

            seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()
            bs = seq_labels.size(0)
            seq_length = seq_labels.size(1)
            
            hidden = torch.zeros((2, bs, 512)).cuda()
            total_loss = 0

            for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm, position=2):
                num_steps += 1
                inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
                pred_cls, pred_trajectory, hidden = model(inputs, hidden)

                cls_loss, reg_loss = loss(pred_cls, pred_trajectory, labels)
                total_loss += (cls_loss + args.mtp_alpha * reg_loss.mean()) / model.module.optimize_per_n_step
            
                if rank == 0 and (num_steps + 1) % args.log_per_n_step == 0:
                    # TODO: add a customized log function
                    writer.add_scalar('train/epoch', epoch, num_steps)
                    writer.add_scalar('loss/cls', cls_loss, num_steps)
                    writer.add_scalar('loss/reg', reg_loss.mean(), num_steps)
                    writer.add_scalar('loss/reg_x', reg_loss[0], num_steps)
                    writer.add_scalar('loss/reg_y', reg_loss[1], num_steps)
                    writer.add_scalar('loss/reg_z', reg_loss[2], num_steps)
                    writer.add_scalar('param/lr', optimizer.param_groups[0]['lr'], num_steps)


                if (t + 1) % model.module.optimize_per_n_step == 0:
                    hidden = hidden.clone().detach()
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
                    optimizer.step()

                    if rank == 0:
                        writer.add_scalar('loss/total', total_loss, num_steps)

                    total_loss = 0



            if not isinstance(total_loss, int):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
                optimizer.step()

                if rank == 0:
                    writer.add_scalar('loss/total', total_loss, num_steps)



        lr_scheduler.step()
        if (epoch + 1) % args.val_per_n_epoch == 0:

            if rank == 0:
                # save model
                ckpt_path = os.path.join(writer.log_dir, 'epoch_%d.pth' % epoch)
                torch.save(model.module.state_dict(), ckpt_path)
                print('[Epoch %d] checkpoint saved at %s' % (epoch, ckpt_path))

            model.eval()

            with torch.no_grad():

                saved_metric_epoch = get_val_metric_keys()

                for batch_idx, data in enumerate(tqdm(val_dataloader, 
                                                      leave=False, disable=disable_tqdm, position=1)):
                    
                    seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()

                    bs = seq_labels.size(0)
                    seq_length = seq_labels.size(1)
                    
                    hidden = torch.zeros((2, bs, 512), device=seq_inputs.device)
                    for t in tqdm(range(seq_length), leave=False, disable=True, position=2):

                        inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]

                        pred_cls, pred_trajectory, hidden = model(inputs, hidden)

                        metrics = get_val_metric(pred_cls, pred_trajectory.view(-1, args.M, args.num_pts, 3), labels)
                        
                        for k, v in metrics.items():
                            saved_metric_epoch[k].append(v.float().mean().item())
                
                dist.barrier()  # Wait for all processes
                # sync
                metric_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')
                counter_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda')

                # From Python 3.6 onwards, the standard dict type maintains insertion order by default.
                # But, programmers should not rely on it.
                for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                    metric_single[i] = np.mean(saved_metric_epoch[k])
                    counter_single[i] = len(saved_metric_epoch[k])

                metric_gather = [torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')[None] for _ in range(world_size)]
                counter_gather = [torch.zeros((len(saved_metric_epoch), ),
                                dtype=torch.int32, device='cuda')[None] for _ in range(world_size)]
                dist.all_gather(metric_gather, metric_single[None])
                dist.all_gather(counter_gather, counter_single[None])

                if rank == 0:
                    metric_gather = torch.cat(metric_gather, dim=0)  # [world_size, num_metric_keys]
                    counter_gather = torch.cat(counter_gather, dim=0)  # [world_size, num_metric_keys]
                    metric_gather_weighted_mean = (metric_gather * counter_gather).sum(0) / counter_gather.sum(0)
                    for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                        writer.add_scalar(k, metric_gather_weighted_mean[i], num_steps)
                dist.barrier()

            model.train()

    cleanup()








if __name__ == "__main__":
    print('[%.2f]' % time.time(), 'starting job...', os.environ['SLURM_PROCID'], 'of', os.environ['SLURM_NTASKS'], flush=True)

    parser = ArgumentParser()
    parser = get_hyperparameters(parser)
    args = parser.parse_args()

    setup(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']))
    main(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']), args=args)













































































# import os
# import time
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow.keras.backend as KB
 
# from modelB6 import get_model


# from tqdm import tqdm

# import glob
# import pickle

# # from serverB6 import client_generator, BATCH_SIZE, STEPS, EPOCHS

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

 
# # print(f"## (len(tf.config.experimental.list_physical_devices('GPU'))): {(len(tf.config.experimental.list_physical_devices('GPU')))}")



# class AttrDict(dict):
#     def __getattr__(self, a):
#         return self[a]

# # para.
# para = AttrDict(
#   {

#     'total_steps': 1000,
#     'base_lr': 1e-3,
#     'decay_type': 'cosine',
#     'warmup_steps': 5,
#     'grad_norm_clip': 1,
#     'lr_min': 1e-4,

#     'accum_steps': 4,
#     'batch_size': 16,

#     'weight_decay': 0.004, # 0.05
#     'ema_momentum': 0.99, #0.99,

#     'host': "localhost",
#     'port': 5557,
#     'port_val': 5558,

#     'log_interval': 5,
#     'save_interval': 25,
#     'validate_interval': 1000,

#     'ckpt_load_path': f'ckpt/modelB6-{2100}.h5',
#   }
# ) 



# def normalize_img(imgs):

 
#     assert imgs.shape[1] == 12

#     c = 12

#     mean = [np.mean(imgs[:, i, ...]) for i in range(c)]
#     std = [np.std(imgs[:, i, ...]) for i in range(c)]
  

#     for i in range(c):
#         imgs[:, i, ...] = (imgs[:, i, ...] - mean[i]) / std[i]

#     return imgs



# def get_data(pkl_files):

#     file_idx = 0
#     while(1):

#         pkl_file = pkl_files[file_idx]
#         # print(f'## get data: {pkl_file}')
#         file_idx += 1
#         if(file_idx >= len(pkl_files)): file_idx = 0
 
            
#         with open(pkl_file, 'rb') as f:      

#             data = pickle.load(f)   

#             n = data['Ximgs'].shape[0]
            
#             assert data['Ximgs'].shape == (n, 12, 128, 256)
#             assert data['Xin1'].shape == (n, 8)
#             assert data['Xin2'].shape == (n, 2)
#             assert data['Xin3'].shape == (n, 512)
            

#             Y_shapes = [385, 386, 386, 58, 200, 200, 200, 8, 4, 32, 12, 512]
#             Y_dim = sum(Y_shapes)

#             for i in range(12):
#                 assert data['Y'][i].shape == (n, Y_shapes[i])

 
            
#             # print(f"bn mean: {np.mean(data['Ximgs'], axis=(0,2,3))}, std: {np.std(data['Ximgs'], axis=(0,2,3))}")

#             data['Ximgs'] = normalize_img(data['Ximgs'])

#             # print(f"an mean: {np.mean(data['Ximgs'], axis=(0,2,3))}, std: {np.std(data['Ximgs'], axis=(0,2,3))}")
#             # exit()

        
#         n_batches = n // para.batch_size
#         for b in range(n_batches):

#             i1 = b * para.batch_size
#             i2 = (b+1) * para.batch_size
#             if(i2 > n): break
             

#             Xins  = [data['Ximgs'][i1:i2], data['Xin1'][i1:i2], data['Xin2'][i1:i2], data['Xin3'][i1:i2]]
#             Xins = [tf.convert_to_tensor(x, dtype=tf.float32) for x in Xins]

#             # Ytrue = np.hstack(tup[4:])
#             Ytrue = [data['Y'][i][i1:i2] for i in range(12)]
#             Ytrue = np.hstack(Ytrue)
#             assert Ytrue.shape == (para.batch_size, Y_dim)
#             Ytrue = tf.convert_to_tensor(Ytrue, dtype=tf.float32)
        
#             # print(f'## yield batch: {b} of file {file_idx}')
#             yield Xins, Ytrue






# def maxae(y_true, y_pred):
#   return KB.max(KB.abs(y_pred - y_true), axis=-1)



# PATH_IDX   = 0      
# LL_IDX     = 385    
# RL_IDX     = 771   
# LEAD_IDX   = 1157 

# def train_loss_fn(y_true, y_pred):
  
#   # y_true = y_true[:, PATH_IDX:   LEAD_IDX]
#   # y_pred = y_pred[:, PATH_IDX:   LEAD_IDX]


#   # loss_CS = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)  # MC4 (Md Case 4)
#   loss_MSE = tf.keras.losses.mse(y_true, y_pred)  # MC5
  
#   # loss = 0.5 * loss_CS + 0.5 * loss_MSE  # MC5
 
#   return loss_MSE






# model = get_model()


# if 'ckpt_load_path' in para:
#     model.load_weights(para.ckpt_load_path)  # for retraining
  
# optimizer = tf.keras.optimizers.AdamW(learning_rate=para.base_lr,
#                   weight_decay=para.weight_decay, ema_momentum=para.ema_momentum, clipvalue=1.0)        

# # train_dataset = get_data(20, para.host, port=para.port)
# # val_dataset = get_data(20, para.host, port=para.port_val)


# pkl_files = glob.glob("/home/richard/dataB6/*/data.pkl")
# assert len(pkl_files) == 3
# print(f'pkl_files: {pkl_files}')


# train_dataset = get_data(pkl_files[0:3])
# # val_dataset = get_data(pkl_files[2:3])


# def validate(n=5):
    
#     print(f'## validating...') 
  
#     losses = 0
#     metrics = 0
#     count = 0
#     for i, batch in tqdm(enumerate(val_dataset)):
#         if i >= n: break

#         X, Y = batch
#         count += len(Y)

#         Y_pred = model(X, training=False) # (128, 10)
#         loss, metric = train_loss_fn(Y, Y_pred)

#         losses += loss.numpy().sum()
#         metrics += metric.numpy().sum()

#     assert count == n * para.batch_size, f'{count} != {n * para.batch_size}'

#     losses = losses / count
#     metrics = metrics / count

#     print(f'val loss: {losses}, val metric: {metrics}, count: {count}')

#     return losses, metrics





# def lr_scheduler(step):

#     lr = para.base_lr

#     progress = (step - para.warmup_steps) / float(para.total_steps - para.warmup_steps)
#     progress = np.clip(progress, 0.0, 1.0)

#     if para.decay_type == 'linear':
#         lr = linear_end + (lr - linear_end) * (1.0 - progress)
#     elif para.decay_type == 'cosine':
#         lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
#     else:
#         raise ValueError(f'Unknown lr type {decay_type}')

#     if para.warmup_steps:
#         lr = lr * np.minimum(1., step / para.warmup_steps)

#     lr = max(lr, para.lr_min)

#     return np.asarray(lr, dtype=np.float32) 


 
# @tf.function
# def _train_step(X_step, Y_step):

#     with tf.GradientTape() as tape:
#         Y_pred = model(X_step, training=True) # (128, 10)
#         loss = train_loss_fn(Y_step, Y_pred)
#         loss = tf.reduce_mean(loss)
         
#     grad = tape.gradient(loss, model.trainable_variables)

#     metric = tf.reduce_mean(maxae(Y_step, Y_pred))

#     return loss, metric, grad





# with open("log.txt", "w") as f: f.write("")

 
# update_counts = 0
# for epoch in range(10000):


#     for i, batch in enumerate(train_dataset): # have ~600 training batches.

#         X, Y = batch
                    
#         if(len(X[0]) < para.batch_size): break


#         total_loss = 0.0
#         total_metric = 0.0

#         accum_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
#         step_size = para.batch_size // para.accum_steps
        
#         for step in range(para.accum_steps): 
          
#             # print(f"## (len(tf.config.experimental.list_physical_devices('GPU'))): {(len(tf.config.experimental.list_physical_devices('GPU')))}")
#             X_step = [a[step*step_size:(step+1)*step_size] for a in X]
#             Y_step = Y[step*step_size:(step+1)*step_size]
 
#             loss, metric, grad = _train_step(X_step, Y_step)
 
#             for i in range(len(accum_gradients)):
                 
#                 accum_gradients[i] += grad[i]
            
             
#             total_loss += loss.numpy() 
#             total_metric += metric.numpy() 

       
#         averaged_gradients = [accum_grad / tf.cast(para.accum_steps, tf.float32) for accum_grad in accum_gradients]
#         # clipped_grads, _ = tf.clip_by_global_norm(averaged_gradients, para.grad_norm_clip)
#         optimizer.apply_gradients(zip(averaged_gradients, model.trainable_variables))

     
#         update_counts += 1
#         lr_next = lr_scheduler(update_counts)
#         tf.keras.backend.set_value(optimizer.learning_rate, lr_next)


#         if update_counts % para.log_interval == 0:
#             log = f'[{update_counts}] train loss: {total_loss / para.accum_steps}' + \
#             f', train metric: {total_metric / para.accum_steps}' + f', lr: {lr_next}' 
#             print(log)
#             with open("log.txt", "a") as f: f.write(log + '\n')

        

#         # if update_counts % para.validate_interval == 0:
#         #     loss, metric = validate()                
        

#         if update_counts % para.save_interval == 0:
#             model.save_weights(f'ckpt/modelB6-{update_counts}.h5')
#             print(f'saved ckpt: ckpt/modelB6-{update_counts}.h5')

  


 