


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
        'M': 5, 
        'batch_size': 1, 
        'accum_batchs': 8,        
        'epochs': 100000,  
        'log_per_n_step': 20, 
        'lr': 0.0001, 
        'mtp_alpha': 1.0,  
        'optimize_per_n_step': 32, 
           
        'num_pts': 192,
        
        'horizon': 128,
        'horizon_val': 128,

        'ckpt': [None, f'ckpt/DD-160.h5'][0],
        
        'save_interval': 20,
        'log_interval': 8,
        'val_interval': 8

    }) 
 

  
transformerEncoder_args = AttrDict({ 
        "heads":12,  
        "depth":4, # transformer's layer number 
        "hidden_size": 768, 
        "dropout_rate": 0.0, 
        "mlp_dim": 3072,
        "seq_len": 100
    })

 