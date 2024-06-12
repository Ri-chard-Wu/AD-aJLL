


import numpy as np

class AttrDict(dict):
    def __getattr__(self, key):
        if key in self:
            value = self[key]
            if isinstance(value, dict):
                return AttrDict(value)
            return value
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")






 

train_args = AttrDict({
       
        'batch_size': 8, 
        'accum_steps': 1,        
        'epochs': 100000,  
         
        'total_steps': 2000,
        'base_lr': 1e-2, 
        'warmup_steps': 100,
        'lr_min': 1e-4,
        
                
        'num_pts': 192,
        
        'horizon': 512,
        'horizon_val': 128,

        'ckpt': [None, f'ckpt/DDViVit-3999.h5'][0],
        
        'save_interval': 500,
        'log_interval': 8,
        'val_interval': 500,
        'log_wandb_interval': 200

    }) 
 

  
transformerEncoder_args = AttrDict({ 
        "heads":8,  
        "depth":2, # transformer's layer number 
        "hidden_size": [512, 1024][1], 
        "dropout_rate": 0.0, 
        "mlp_dim": 1024,
        "seq_len": 32
    })

 