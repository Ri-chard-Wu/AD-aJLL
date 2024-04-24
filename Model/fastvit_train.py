

import math
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras import datasets

import numpy as np

import pickle  
import os
# import glob
# print(f'glob.glob(demo_dir + "*.mjl"): {glob.glob("./*")}')


# exit()
 

class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]


 
class ModuleList(tf.keras.layers.Layer):
    def __init__(self, name, seq_maker):
        super(ModuleList, self).__init__(name=name)
        self.seq_maker = seq_maker

    def build(self, input_shape):
        self.seq = self.seq_maker()


    def call(self, x, training=True):
        for m in self.seq:
            x = m(x, training=training)
        return x




# layernorm para name tf -> torch: gamma -> scale, beta->bias.
class BatchNorm2d(tf.keras.layers.Layer):
    def __init__(self, momentum=0.99, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='weight', shape=input_shape[-1], initializer='ones')
        self.beta = self.add_weight(name='bias', shape=input_shape[-1], initializer='zeros')
        self.moving_mean = self.add_weight(name='running_mean', shape=input_shape[-1], initializer='zeros', trainable=False)
        self.moving_variance = self.add_weight(name='running_var', shape=input_shape[-1], initializer='ones', trainable=False)
        self.num_batches_tracked = self.add_weight(name='num_batches_tracked', shape=(), initializer='ones', trainable=False)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=True):
        if training:
            batch_mean, batch_variance = \
                tf.nn.moments(inputs, axes=list(range(len(inputs.shape) - 1)), keepdims=False)

            self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean)
            self.moving_variance.assign(self.momentum * self.moving_variance + (1 - self.momentum) * batch_variance)

            return tf.nn.batch_normalization(inputs, mean=batch_mean, variance=batch_variance,
                                offset=self.beta, scale=self.gamma, variance_epsilon=self.epsilon)
        else:
            return tf.nn.batch_normalization(inputs, mean=self.moving_mean, variance=self.moving_variance,
                                             offset=self.beta, scale=self.gamma, variance_epsilon=self.epsilon)




class DropPathTF(tf.keras.layers.Layer):

    def __init__(self, drop_prob, **kwargs):
        super(DropPathTF, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs, training=None):

        if not training or self.drop_prob == 0: # yes, drop_prob == 0.
            return inputs

        batch_size = tf.shape(inputs)[0]
        keep_prob = 1 - self.drop_prob

        # Generate random numbers for each example in the batch
        random_tensor = keep_prob + tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)

        # Convert the random tensor to a binary mask
        binary_mask = tf.floor(random_tensor)

        # Scale the inputs to compensate for dropped paths
        output = tf.divide(inputs, keep_prob) * binary_mask

        return output


class ReparamLargeKernelConv(tf.keras.layers.Layer):

    DEFAULT_CFG = {
        'inference_mode': False,
        'activation': tf.nn.gelu
    }

    def __init__(self, name, cfg):
        self.cfg = cfg
        for key, val in self.DEFAULT_CFG.items():
            if key in self.cfg: continue
            self.cfg[key] = val
        super(ReparamLargeKernelConv, self).__init__(name=name)

    def build(self, input_shape):

        cfg = self.cfg
        self.activation = cfg.activation
        self.padding = cfg.kernel_size // 2

        self.lkb_origin = self._conv_bn('lkb_origin', kernel_size=cfg.kernel_size, padding=self.padding)
        assert cfg.small_kernel <= cfg.kernel_size
        self.small_conv = self._conv_bn('small_conv', kernel_size=cfg.small_kernel, padding=cfg.small_kernel // 2)


    def _conv_bn(self, name, kernel_size, padding):

        cfg = self.cfg

        def seq_maker():
            seq = []
            seq.append(tf.keras.layers.ZeroPadding2D(padding=(padding, padding)))
            seq.append(
                tf.keras.layers.Conv2D(
                                    filters=cfg.out_channels,
                                    kernel_size=kernel_size,
                                    strides=cfg.stride,
                                    # padding=padding,
                                    groups=cfg.groups,
                                    use_bias=False,
                                    name='conv'
                                )
                )

            seq.append(BatchNorm2d(name='bn'))
            return seq

        # return tf.keras.Sequential(seq, name=name)
        return ModuleList(name, seq_maker)




    def call(self, x, training=True):

        out = self.lkb_origin(x, training=training)
        out += self.small_conv(x, training=training)

        return self.activation(out)




 


class SEBlock(tf.keras.layers.Layer):

    def __init__(self, name, in_channels, rd_ratio = 0.0625):
        self.in_channels = in_channels
        self.rd_ratio = rd_ratio
        super(SEBlock, self).__init__(name=name)

    def build(self, input_shape):

        self.reduce = tf.keras.layers.Conv2D(
                    filters=int(self.in_channels * self.rd_ratio),
                    kernel_size=1,
                    strides=1,
                    use_bias=True,
                    name='reduce'
                  )

        self.expand = tf.keras.layers.Conv2D(
                    filters=self.in_channels,
                    kernel_size=1,
                    strides=1,
                    use_bias=True,
                    name='expand'
                  )

    # def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    def call(self, inputs, training=True):

        # inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])

        b, h, w, c = inputs.shape

        # x = self.avg_pool2d(inputs, kernel_size=[h, w])
        x = tf.nn.avg_pool2d(inputs, ksize=[h, w], strides=1, padding='VALID')

        x = self.reduce(x)

        # x = F.relu(x)
        x = tf.nn.relu(x)

        x = self.expand(x)

        # x = torch.sigmoid(x)
        x = tf.math.sigmoid(x)


        # assert list(x.shape) == [1, 1, 1, 1216], list(x.shape)
        # x = x.view(-1, c, 1, 1)

        return inputs * x





class MobileOneBlock(tf.keras.layers.Layer):


    DEFAULT_CFG = {

        'stride': 1,
        'padding': 0,
        'dilation': 1,
        'groups': 1,
        'inference_mode': False,
        'use_se': False,
        'use_act': True,
        'use_scale_branch': True,
        'num_conv_branches': 1,
        'activation': tf.nn.gelu
    }

    def __init__(self, name, cfg):
        self.cfg = cfg
        for key, val in self.DEFAULT_CFG.items():
            if key in self.cfg: continue
            self.cfg[key] = val

        super(MobileOneBlock, self).__init__(name=name)


    def build(self, input_shape):

        cfg = self.cfg

        # Check if SE-ReLU is requested
        if cfg.use_se:
            self.se = SEBlock('se', cfg.out_channels)
        else:
            self.se = tf.keras.layers.Identity()

        if cfg.use_act:
            self.activation = cfg.activation
        else:
            self.activation = tf.keras.layers.Identity()

        # Re-parameterizable skip connection
        self.rbr_skip = (
            BatchNorm2d(name='rbr_skip')
            if cfg.out_channels == cfg.in_channels and cfg.stride == 1
            else None
        )

        # Re-parameterizable conv branches
        if cfg.num_conv_branches > 0:
            rbr_conv = list()
            for _ in range(cfg.num_conv_branches):
              # nn.Conv2d + bachnorm.
                rbr_conv.append(self._conv_bn(f'rbr_conv/{_}', kernel_size=cfg.kernel_size, padding=cfg.padding))

            # self.rbr_conv = nn.ModuleList(rbr_conv)
            self.rbr_conv = rbr_conv

        else:
            self.rbr_conv = None

        # Re-parameterizable scale branch
        self.rbr_scale = None
        if (cfg.kernel_size > 1) and cfg.use_scale_branch:
            self.rbr_scale = self._conv_bn('rbr_scale', kernel_size=1, padding=0)



    def _conv_bn(self, name, kernel_size, padding):

        cfg = self.cfg

        def seq_maker():

            seq = []
            assert padding in [0, 1], padding

            seq.append(
                tf.keras.layers.Conv2D(
                                    filters=cfg.out_channels,
                                    kernel_size=kernel_size,
                                    strides=cfg.stride,
                                    padding=['valid', 'same'][padding],
                                    groups=cfg.groups,
                                    use_bias=False,
                                    name='conv'
                                )
                )

            seq.append(BatchNorm2d(name='bn'))

            return seq

        return ModuleList(name, seq_maker)



    def call(self, x, training=True):

        cfg = self.cfg


        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x, training=training)


        # Scale branch output
        scale_out = 0
        # print(f'self.rbr_scale: {self.rbr_scale}')
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x, training=training)


        # Other branches
        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(cfg.num_conv_branches):
                out += self.rbr_conv[ix](x, training=training)


        return self.activation(self.se(out))



def convolutional_stem(in_channels, out_channels, inference_mode=False):

    def seq_maker():
        seq = []

        seq.append(
            MobileOneBlock(
                '0',
                AttrDict({
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'kernel_size': 3,
                    'stride': 2,
                    'padding': 1,
                    'groups': 1,
                    'inference_mode': inference_mode,
                    'use_se': False,
                    'num_conv_branches': 1
                })
            )
        )

        seq.append(
            MobileOneBlock(
                '1',
                AttrDict({'in_channels': out_channels,
                'out_channels': out_channels,
                'kernel_size': 3,
                'stride': 2,
                'padding': 1,
                'groups': out_channels,
                'inference_mode': inference_mode,
                'use_se': False,
                'num_conv_branches': 1})
            )
        )

        seq.append(

            MobileOneBlock(
                '2',
                AttrDict({'in_channels': out_channels,
                'out_channels': out_channels,
                'kernel_size': 1,
                'stride': 1,
                'padding': 0,
                'groups': 1,
                'inference_mode': inference_mode,
                'use_se': False,
                'num_conv_branches': 1})
            )
        )

        return  seq


    return ModuleList('patch_embed', seq_maker)



class MHSA(tf.keras.layers.Layer):

    DEFAULT_CFG = {
        'head_dim': 32,
        'qkv_bias': False,
        'attn_drop': 0.0,
        'proj_drop': 0.0,
    }

    def __init__(self, name, cfg):
        self.cfg = cfg
        for key, val in self.DEFAULT_CFG.items():
            if key in self.cfg: continue
            self.cfg[key] = val

        super(MHSA, self).__init__(name=name)

    def build(self, input_shape):

        cfg = self.cfg


        # print(f'cfg.dim: {cfg.dim}, cfg.head_dim: {cfg.head_dim}')

        assert cfg.dim % cfg.head_dim == 0, "dim should be divisible by head_dim"



        self.num_heads = cfg.dim // cfg.head_dim
        self.scale = cfg.head_dim**-0.5


        self.qkv = tf.keras.layers.Dense(cfg.dim * 3, use_bias=cfg.qkv_bias, name='qkv')
        self.attn_drop = tf.keras.layers.Dropout(cfg.attn_drop)
        self.proj = tf.keras.layers.Dense(cfg.dim, name='proj')
        self.proj_drop = tf.keras.layers.Dropout(cfg.proj_drop)


    def call(self, x, training=True):

        cfg = self.cfg



        shape = x.shape
        B, H, W, C = shape
        N = H * W
        if len(shape) == 4:
            # x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
            # x = Rearrange('b h w c -> b (h w) c')(x)
            x = tf.reshape(x, (B, H * W, C))

        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, N, 3, self.num_heads, self.head_dim)
        #     .permute(2, 0, 3, 1, 4) # qvk, B, self.num_heads, N, self.head_dim
        # )

        x = self.qkv(x)
        x = tf.reshape(x, (B, N, 3, self.num_heads, cfg.head_dim))

        # (3, 1, 19, 64, 32)
        x = tf.transpose(x, perm=[2, 0, 3, 1, 4]) # qvk, B, num_heads, N, head_dim

        # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v = x # (b h p d)

        q = x[0]
        k = x[1]
        v = x[2]

        # [5, 19, 64, 32], [5, 19, 32, 64] -> [5, 19, 64, 64]
        # attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = tf.einsum('bpij,bpkj->bpik', q * self.scale, k)

        # attn = attn.softmax(dim=-1)
        attn = tf.nn.softmax(attn, axis=-1) # [5, 19, 64, 64]


        attn = self.attn_drop(attn, training=training) # (b h p p) == (1, 19, 64, 64)

        # v.shape: [5, 19, 64, 32]
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # (1, 19, 64, 32)
        x = tf.einsum('bpij,bpjk->bpik', attn, v) # (b h p p) (b h p d) -> (b h p d)


        x = tf.transpose(x, perm=[0, 2, 1, 3]) # (b p h d)

        # print(f'[c] x.shape: {x.shape}')

        x = tf.reshape(x, (B, N, C)) # (b p h*d) == (b p c)

        x = self.proj(x)  # (b p dim)
        x = self.proj_drop(x, training=training)
        if len(shape) == 4:
            # x = x.transpose(-2, -1).reshape(B, C, H, W)
            x = tf.reshape(x, (B, H, W, C))

        return x


class PatchEmbed(tf.keras.layers.Layer):
    """Convolutional patch embedding layer."""

    DEFAULT_CFG = {
        'inference_mode': False
    }

    def __init__(self, name, cfg):
        self.cfg = cfg
        for key, val in self.DEFAULT_CFG.items():
            if key in self.cfg: continue
            self.cfg[key] = val

        super(PatchEmbed, self).__init__(name=name)


    def build(self, input_shape):

        cfg = self.cfg

        def seq_maker():
            block = list()

            block.append(
                ReparamLargeKernelConv(
                    '0',
                    AttrDict({'in_channels': cfg.in_channels,
                    'out_channels': cfg.embed_dim,
                    'kernel_size': cfg.patch_size, # 7
                    'stride': cfg.stride,
                    'groups': cfg.in_channels,
                    'small_kernel': 3,
                    'inference_mode': cfg.inference_mode})
                )
            )

            block.append(
                MobileOneBlock(
                    '1',
                    AttrDict({'in_channels': cfg.embed_dim,
                    'out_channels': cfg.embed_dim,
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'groups': 1,
                    'inference_mode': cfg.inference_mode,
                    'use_se': False,
                    'num_conv_branches': 1})
                )
            )
            return block
        # self.proj = nn.Sequential(*block)

        self.proj = ModuleList('proj', seq_maker)

    def call(self, x, training=True):
        x = self.proj(x, training=training)
        return x



class ConvFFN(tf.keras.layers.Layer):

    DEFAULT_CFG = {
        'hidden_channels': None,
        'out_channels': None,
        'act_layer': tf.nn.gelu,
        'drop': 0.0
    }

    def __init__(self, name, cfg):
        self.cfg = cfg
        for key, val in self.DEFAULT_CFG.items():
            if key in self.cfg: continue
            self.cfg[key] = val

        super(ConvFFN, self).__init__(name=name)


    def build(self, input_shape):

        cfg = self.cfg

        cfg.out_channels = cfg.out_channels or cfg.in_channels
        cfg.hidden_channels = cfg.hidden_channels or cfg.in_channels


        seq = []


        seq.append(tf.keras.layers.ZeroPadding2D(padding=(3, 3)))
        seq.append(
            tf.keras.layers.Conv2D(
                                filters=cfg.out_channels,
                                kernel_size=7,
                                # padding=3,
                                groups=cfg.in_channels,
                                use_bias=False,
                                name='conv/conv'
                              )
            )
        seq.append(BatchNorm2d(name='conv/bn'))
        # self.conv = tf.keras.Sequential(seq)
        self.conv = seq



        # self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.fc1 = tf.keras.layers.Conv2D(filters=cfg.hidden_channels, kernel_size=1, name='fc1')

        self.act = tf.nn.gelu

        # self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.fc2 = tf.keras.layers.Conv2D(filters=cfg.out_channels, kernel_size=1, name='fc2')

        # self.drop = nn.Dropout(drop)
        self.drop = tf.keras.layers.Dropout(cfg.drop)

    def call(self, x, training=True):

        for b in self.conv:
            x = b(x, training=training)


        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x



class RepCPE(tf.keras.layers.Layer):

    def __init__(
          self,
          name,
          in_channels,
          embed_dim = 768,
          spatial_shape = (7, 7),
          inference_mode = False
        ):

        # print(f'name: {name}')

        self.cfg = AttrDict({
            'spatial_shape': spatial_shape,
            'embed_dim': embed_dim
        })

        # for key, val in self.DEFAULT_CFG.items():
        #     if key in self.cfg: continue
        #     self.cfg[key] = val
        super(RepCPE, self).__init__(name=name)


    def build(self, input_shape):

        cfg = self.cfg

        if isinstance(cfg.spatial_shape, int):
            cfg.spatial_shape = tuple([cfg.spatial_shape] * 2)

        assert isinstance(cfg.spatial_shape, Tuple)
        assert len(cfg.spatial_shape) == 2


        padding = int(cfg.spatial_shape[0] // 2)

        # def seq_maker():
        #     return [
        #            tf.keras.layers.ZeroPadding2D(padding=(padding, padding)),
        #            tf.keras.layers.Conv2D(
        #                         filters=cfg.embed_dim,
        #                         kernel_size=cfg.spatial_shape,
        #                         strides=1,
        #                         # padding=int(cfg.spatial_shape[0] // 2),
        #                         groups=cfg.embed_dim,
        #                         use_bias=True,
        #                         name='pe'
        #                       )
        #            ]

        # self.pe = ModuleList('', seq_maker)

        self.pe_pad = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))
        self.pe = tf.keras.layers.Conv2D(
                                filters=cfg.embed_dim,
                                kernel_size=cfg.spatial_shape,
                                strides=1,
                                # padding=int(cfg.spatial_shape[0] // 2),
                                groups=cfg.embed_dim,
                                use_bias=True,
                                name='pe'
                              )



    def call(self, x, training=True):

        # print(f'conv shape: {self.pe.weights[0].shape}, x.shape: {x.shape}')
        x = self.pe(self.pe_pad(x)) + x
        return x



class RepMixer(tf.keras.layers.Layer):

    DEFAULT_CFG = {
        'kernel_size': 3, # 3
        'use_layer_scale': True, # True
        'layer_scale_init_value': 1e-5,
        'inference_mode':  False,
    }


    def __init__(self, name, cfg):
        self.cfg = cfg
        for key, val in self.DEFAULT_CFG.items():
            if key in self.cfg: continue
            self.cfg[key] = val

        super(RepMixer, self).__init__(name=name)



    def build(self, input_shape):

        cfg = self.cfg


        self.norm =  MobileOneBlock(
            'norm',
            AttrDict({'in_channels': cfg.dim,
            'out_channels': cfg.dim,
            'kernel_size': cfg.kernel_size, # 3
            'padding': cfg.kernel_size // 2,
            'groups': cfg.dim,
            'use_act': False,
            'use_scale_branch': False,
            'num_conv_branches': 0})
        )
        self.mixer =  MobileOneBlock(
            'mixer',
            AttrDict({'in_channels': cfg.dim,
            'out_channels': cfg.dim,
            'kernel_size': cfg.kernel_size, # 3
            'padding': cfg.kernel_size // 2,
            'groups': cfg.dim,
            'use_act': False})
        )

        # self.layer_scale = nn.Parameter(
        #     layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
        # )
        self.layer_scale = self.add_weight("layer_scale", shape=[1, 1, cfg.dim], dtype=tf.float32)


    def call(self, x, training=True):

        x = x + self.layer_scale * (self.mixer(x, training=training) - self.norm(x, training=training))

        return x







class RepMixerBlock(tf.keras.layers.Layer):


    DEFAULT_CFG=  {
        'kernel_size': 3,
        'mlp_ratio': 4.0,
        'act_layer': tf.nn.gelu,
        'drop': 0.0,
        'drop_path': 0.0,
        'use_layer_scale': True, # True
        'layer_scale_init_value': 1e-5,
        'inference_mode': False,
    }


    def __init__(self, name, cfg):
        self.cfg = cfg
        for key, val in self.DEFAULT_CFG.items():
            if key in self.cfg: continue
            self.cfg[key] = val

        super(RepMixerBlock, self).__init__(name=name)





    def build(self, input_shape):

        cfg = self.cfg

        self.token_mixer = RepMixer(
            'token_mixer',
            AttrDict({'dim': cfg.dim,
            'kernel_size': cfg.kernel_size, # 3
            'use_layer_scale': cfg.use_layer_scale, # True
            'layer_scale_init_value': cfg.layer_scale_init_value,
            'inference_mode': cfg.inference_mode})
        )

        assert cfg.mlp_ratio > 0

        mlp_hidden_dim = int(cfg.dim * cfg.mlp_ratio)
        self.convffn = ConvFFN(
            'convffn',
            AttrDict({'in_channels': cfg.dim,
            'hidden_channels': mlp_hidden_dim,
            'act_layer': cfg.act_layer,
            'drop': cfg.drop})
        )


        # Drop Path
        self.drop_path = DropPathTF(cfg.drop_path) if cfg.drop_path > 0.0 else tf.keras.layers.Identity()

        # self.layer_scale = nn.Parameter(
        #     cfg.layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
        # )
        self.layer_scale = self.add_weight("layer_scale", shape=[1, 1, cfg.dim], dtype=tf.float32)


    def call(self, x, training=True):

        x = self.token_mixer(x, training=training)
        x = x + self.drop_path(self.layer_scale * self.convffn(x, training=training), training=training)

        return x


class AttentionBlock(tf.keras.layers.Layer):


    DEFAULT_CFG =  {
        'mlp_ratio': 4.0,
        'act_layer': tf.nn.gelu,
        'norm_layer': BatchNorm2d,
        'drop': 0.0,
        'drop_path': 0.0,
        'use_layer_scale': True,
        'layer_scale_init_value': 1e-5,
    }


    def __init__(self, name, cfg):
        self.cfg = cfg
        for key, val in self.DEFAULT_CFG.items():
            if key in self.cfg: continue
            self.cfg[key] = val
        super(AttentionBlock, self).__init__(name=name)



    def build(self, input_shape):

        cfg = self.cfg

        # self.norm = cfg.norm_layer(cfg.dim)
        self.norm = cfg.norm_layer(name='norm')
        self.token_mixer =  MHSA('token_mixer', AttrDict({'dim': cfg.dim}))

        assert cfg.mlp_ratio > 0


        mlp_hidden_dim = int(cfg.dim * cfg.mlp_ratio)
        self.convffn = ConvFFN(
            'convffn',
            AttrDict({'in_channels': cfg.dim,
            'hidden_channels': mlp_hidden_dim,
            'act_layer': cfg.act_layer,
            'drop': cfg.drop})
        )

        # Drop path
        self.drop_path = DropPathTF(cfg.drop_path) if cfg.drop_path > 0.0 else tf.keras.layers.Identity()

        # Layer Scale
        self.use_layer_scale = cfg.use_layer_scale

        self.layer_scale_1 = self.add_weight("layer_scale_1", shape=[1, 1, cfg.dim], dtype=tf.float32)
        self.layer_scale_2 = self.add_weight("layer_scale_2", shape=[1, 1, cfg.dim], dtype=tf.float32)


    def call(self, x, training=True):

        x1 = self.norm(x, training=training)

        x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x1, training=training), training=training)

        x = x + self.drop_path(self.layer_scale_2 * self.convffn(x, training=training), training=training)

        return x



def basic_blocks(
        name,
        dim,
        block_index,
        num_blocks,
        token_mixer_type,
        kernel_size = 3, # 3
        mlp_ratio = 4.0,
        act_layer = tf.nn.gelu,
        norm_layer = BatchNorm2d,
        drop_rate = 0.0,
        drop_path_rate = 0.0,
        use_layer_scale = True, # True
        layer_scale_init_value = 1e-5,
        inference_mode = False
    ):



    def seq_maker():

        blocks = []
        for block_idx in range(num_blocks[block_index]):


            block_dpr = (
                drop_path_rate * (block_idx + sum(num_blocks[:block_index])) / (sum(num_blocks) - 1)
            )

            if token_mixer_type == "repmixer":

                blocks.append(
                    RepMixerBlock(
                        f'{len(blocks)}',
                        AttrDict({'dim': dim,
                        'kernel_size': kernel_size, # 3
                        'mlp_ratio': mlp_ratio,
                        'act_layer': act_layer,
                        'drop': drop_rate,
                        'drop_path': block_dpr,
                        'use_layer_scale': use_layer_scale, # True
                        'layer_scale_init_value': layer_scale_init_value,
                        'inference_mode': inference_mode})
                    )
                )

            elif token_mixer_type == "attention":

                blocks.append(
                    AttentionBlock(
                        f'{len(blocks)}',
                        AttrDict({'dim': dim,
                        'mlp_ratio': mlp_ratio,
                        'act_layer': act_layer,
                        'norm_layer': norm_layer,
                        'drop': drop_rate,
                        'drop_path': block_dpr,
                        'use_layer_scale': use_layer_scale, # True
                        'layer_scale_init_value': layer_scale_init_value})
                    )
                )
            else:
                raise ValueError(
                    "Token mixer type: {} not supported".format(token_mixer_type)
                )


        return blocks

    blocks = ModuleList(name, seq_maker)

    return blocks




# class FastViT(tf.keras.layers.Layer):

class FastViT(tf.keras.Model):

    DEFAULT_CFG =  {
        'embed_dims': None, #
        'mlp_ratios': None, #
        'downsamples': None, #

        'repmixer_kernel_size': 3,

        'act_layer' :  tf.nn.gelu,
        'norm_layer' :  BatchNorm2d,

        'num_classes': 10,

        'pos_embs': None, #

        'down_patch_size': 7,
        'down_stride': 2,
        'drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'use_layer_scale': True,
        'layer_scale_init_value': 1e-5,
        'fork_feat': False,
        'init_cfg': None,
        'pretrained': None,
        'cls_ratio': 2.0,
        'inference_mode': False,
    }



    def __init__(self, cfg):
        self.cfg = cfg
        for key, val in self.DEFAULT_CFG.items():
            if key in self.cfg: continue
            self.cfg[key] = val
        super(FastViT, self).__init__(name='')



        # def build(self, input_shape):
        # cfg = self.cfg

        self.patch_embed = convolutional_stem(3, cfg.embed_dims[0], cfg.inference_mode)


        # network[0]: basic_blocks
        # network[1]: PatchEmbed
        # network[2]: basic_blocks
        # network[3]: PatchEmbed
        # network[4]: basic_blocks
        # network[5]: PatchEmbed
        # network[6]: pos_embs
        # network[7]: basic_blocks

        def seq_maker():
            network = []
            for i in range(len(cfg.layers)):

                # Add position embeddings if requested
                if cfg.pos_embs[i] is not None:
                    network.append(
                        cfg.pos_embs[i](
                            f'{len(network)}', cfg.embed_dims[i], cfg.embed_dims[i], inference_mode=cfg.inference_mode
                        )
                    )


                network.append(
                    basic_blocks(
                        f'{len(network)}',
                        cfg.embed_dims[i],
                        i,
                        cfg.layers,
                        token_mixer_type=cfg.token_mixers[i],
                        kernel_size=cfg.repmixer_kernel_size, # 3
                        mlp_ratio=cfg.mlp_ratios[i],
                        act_layer=cfg.act_layer,
                        norm_layer=cfg.norm_layer,
                        drop_rate=cfg.drop_rate,
                        drop_path_rate=cfg.drop_path_rate,
                        use_layer_scale=cfg.use_layer_scale,
                        layer_scale_init_value=cfg.layer_scale_init_value,
                        inference_mode=cfg.inference_mode,
                    )
                )

                if i >= len(cfg.layers) - 1: break

                # Patch merging/downsampling between stages.
                if cfg.downsamples[i] or cfg.embed_dims[i] != cfg.embed_dims[i + 1]:
                    network.append(
                        PatchEmbed(
                            f'{len(network)}',
                            AttrDict({'patch_size': cfg.down_patch_size, # 7
                            'stride': cfg.down_stride,
                            'in_channels': cfg.embed_dims[i],
                            'embed_dim': cfg.embed_dims[i + 1],
                            'inference_mode': cfg.inference_mode})
                        )
                    )
            return network

        self.network = ModuleList(f'network', seq_maker)

        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        self.conv_exp = MobileOneBlock(
            'conv_exp',
            AttrDict({'in_channels': cfg.embed_dims[-1],
            'out_channels': int(cfg.embed_dims[-1] * cfg.cls_ratio),
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'groups': cfg.embed_dims[-1],
            'inference_mode': cfg.inference_mode,
            'use_se': True,
            'num_conv_branches': 1})
        )

        # self.head = nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes)
        self.head = tf.keras.layers.Dense(cfg.num_classes, name='head')




    def save_ckpt(self, dir_name, name): 

        weights = {}

        for w in self.weights:
            weights[w.name] = w


        path = os.path.join(dir_name, name)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

        print(f'saved ckpt: {path}')
        

    def load_ckpt(self, dir_name, name):
        
        path = os.path.join(dir_name, name)
        with open(path, 'rb') as f:            
            weights = pickle.load(f)
        
        

        for i, w in enumerate(self.weights):
            # weights[w.name] = w
            assert w.name in weights
            w.assign(weights[w.name])
            # print(f'[{i}] type(weights[w.name]): {type(weights[w.name])}, w.name: {w.name}, weights[w.name].shape: {weights[w.name].shape}')

        print(f'loaded ckpt: {path}')


    # @tf.function
    # def call(self, x, training=True):


    #     x = self.patch_embed(x, training=training)

    #     # for idx, block in enumerate(self.network):
    #     #     x = block(x)

    #     x = self.network(x, training=training)

    #     x = self.conv_exp(x, training=training) # (1, 8, 8, 1216)

    #     x = self.gap(x, training=training) # (1, 1216)
 
    #     return x



    @tf.function
    def call(self, x, training=True):


        x = self.patch_embed(x, training=training)

        # for idx, block in enumerate(self.network):
        #     x = block(x)

        x = self.network(x, training=training)

        x = self.conv_exp(x, training=training) # (1, 8, 8, 1216)

        x = self.gap(x, training=training) # (1, 1216)

        # x = x.view(x.size(0), -1) # [5, 1216]
        # x = tf.squeeze(x) # [5, 1216]

        cls_out = self.head(x)
        return cls_out


layers = [2, 2, 6, 2]
embed_dims = [64, 128, 256, 512]
mlp_ratios = [3, 3, 3, 3]
downsamples = [True, True, True, True]
pos_embs = [None, None, None, None]
token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
fastvit = FastViT(
      AttrDict({'layers': layers,
      'embed_dims': embed_dims,
      'token_mixers': token_mixers,
      'pos_embs': pos_embs,
      'mlp_ratios': mlp_ratios,
      'downsamples': downsamples})
)




# layers = [6, 6, 18, 6]
# embed_dims = [76, 152, 304, 608]
# mlp_ratios = [4, 4, 4, 4]
# downsamples = [True, True, True, True]
# pos_embs = [None, None, None, RepCPE]
# token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
# fastvit =  FastViT(
#     AttrDict({'layers': layers,
#     'embed_dims': embed_dims,
#     'token_mixers': token_mixers,
#     'pos_embs': pos_embs,
#     'mlp_ratios': mlp_ratios,
#     'downsamples': downsamples,
#     'layer_scale_init_value': 1e-6})
# )


fastvit(tf.random.uniform([1, 256, 256, 3]))


# fastvit.load_weights('./ckpt/fastvit-tf-finetune-gpu-loss4-acc0p85.h5')
# fastvit.load_weights('./ckpt/acc0p96.h5')
# fastvit.save_ckpt('ckpt', 'acc0p96.pkl')
fastvit.load_ckpt('ckpt', 'acc0p96.pkl')



(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()



# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

num_classes = 10
def preprocess_function(img, label):

    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, [256, 256])
    # img = img / 255.0
    # img = (img - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD

    img = (img - 127.5) / 127.5

    label = tf.one_hot(label, num_classes)
    label = tf.squeeze(label, axis=0)

    return img, label


batch_size = 256
accum_steps = 32

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.map(preprocess_function).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.map(preprocess_function).batch(4)




def validate(n=25):
    """Returns accuracy evaluated on the test set."""

    good = total = 0
    # steps = input_pipeline.get_dataset_info(dataset, 'test')['num_examples'] // batch_size

    # for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
    for i, batch in tqdm(enumerate(test_dataset)):
        if i >= n: break

        x, y = batch

        logits = fastvit(x, training=False)
    #         prob = tf.nn.softmax(logits, axis=-1)
    #         loss = tf.reduce_mean(self.cce(Y, prob))

        y_pred = tf.argmax(logits, axis=-1) # (b,)
        y_true = tf.argmax(y, axis=-1) # (b,)


        # predicted = vit(x)
        is_same = (y_pred == y_true).numpy()
        good += is_same.sum()
        total += len(is_same.flatten())

    return good,  total




good, total = validate(100)
print(f'acc: {good / total}, good: {good}, total: {total}')

exit()



total_steps = 400
base_lr = 5e-4
decay_type = 'cosine'
warmup_steps = 5
grad_norm_clip = 1
linear_end=1e-5


optimizer = tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=0.05, ema_momentum=0.9)
model = fastvit



def lr_scheduler(step):

    lr = base_lr

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = np.clip(progress, 0.0, 1.0)

    if decay_type == 'linear':
        lr = linear_end + (lr - linear_end) * (1.0 - progress)
    elif decay_type == 'cosine':
        lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    else:
        raise ValueError(f'Unknown lr type {decay_type}')

    if warmup_steps:
        lr = lr * np.minimum(1., step / warmup_steps)

    return np.asarray(lr, dtype=np.float32)



train_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                    reduction=tf.keras.losses.Reduction.NONE)


@tf.function
def _train_step(X_step, Y_step):

    # X_step, Y_step = batch_step

    with tf.GradientTape() as tape:
        logits = model(X_step, training=True) # (128, 10)
        Y_pred = tf.nn.softmax(logits, axis=-1)
        loss = train_loss_fn(Y_step, Y_pred)
        loss = tf.reduce_mean(loss)

    grad = tape.gradient(loss, model.trainable_variables)
    #     prob = tf.nn.softmax(logits, axis=-1)
    #     loss = tf.reduce_mean(self.cce(Y_step, prob))

    # grad = tape.gradient(loss, self.model.trainable_variables)

    return loss, grad





try:

    
    log_interval = 5
    save_interval = 15
    validate_interval = 15

    update_counts = 0
    for epoch in range(100):

        # losses = []
        for i, batch in enumerate(train_dataset):
 
            X, Y = batch
            
            if(len(X) < batch_size):
                print(f'len(X) < batch_size: {len(X)}')
                break

            total_loss = 0.0
            accum_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
            step_size = batch_size // accum_steps
            for step in range(accum_steps):
                # with tf.GradientTape() as tape:
                #     Y_pred = model(X, training=True)
                #     loss = train_loss_fn(Y_pred, Y)
                # grad = tape.gradient(loss, model.trainable_variables)

                X_step = X[step*step_size:(step+1)*step_size]
                Y_step = Y[step*step_size:(step+1)*step_size]
                loss, grad = _train_step(X_step, Y_step)

                for i in range(len(accum_gradients)):
                    accum_gradients[i] += grad[i]

                total_loss += loss

            averaged_gradients = [accum_grad / tf.cast(accum_steps, tf.float32) for accum_grad in accum_gradients]
            clipped_grads, _ = tf.clip_by_global_norm(averaged_gradients, grad_norm_clip)
            optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))


            # optimizer.apply_gradients(zip(grad, model.trainable_variables))

            update_counts += 1
            lr_next = lr_scheduler(update_counts)
            tf.keras.backend.set_value(optimizer.learning_rate, lr_next)

            # losses.append(tf.reduce_mean(loss).numpy())

            

            if update_counts % log_interval == 0:
                print(f'[{update_counts}] train_loss: {total_loss.numpy()}, lr: {lr_next}')


            if update_counts % validate_interval == 0:
                good, total = validate()
                print(f'acc: {good / total}, good: {good}, total: {total}')
            

            if update_counts % save_interval == 0:
                model.save_weights(f'ckpt/ckpt-{update_counts}.h5')

   

except KeyboardInterrupt:
    print('KeyboardInterrupt')
     