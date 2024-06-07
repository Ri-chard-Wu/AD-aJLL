


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
        'accum_steps': 8,        
        'epochs': 100000,  
        
        'lr': 0.0001,   
        'num_pts': 192,
        
        'horizon': 1024,
        'horizon_val': 128,

        'ckpt': [None, f'ckpt/DD-1700.h5'][1],
        
        'save_interval': 100,
        'log_interval': 8,
        'val_interval': 100

    }) 
 

  
transformerEncoder_args = AttrDict({ 
        "heads":8,  
        "depth":2, # transformer's layer number 
        "hidden_size": 512, 
        "dropout_rate": 0.0, 
        "mlp_dim": 1024,
        "seq_len": 16
    })

 