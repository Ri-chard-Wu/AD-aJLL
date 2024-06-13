


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import math
import string
import collections 
import tensorflow as tf

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])


BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out', 
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

 

def get_swish(**kwargs):
 
    def swish(x): 
        return x * tf.keras.backend.sigmoid(x)

    return swish


def get_dropout(**kwargs):
  
    class FixedDropout(tf.keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = tf.keras.backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout


def round_filters(filters, width_coefficient, depth_divisor):
 
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)

    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor

    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier.""" 
    return int(math.ceil(depth_coefficient * repeats))

 



def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix='', ):
    """Mobile Inverted Residual Bottleneck."""

    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    # workaround over non working dropout with None in noise_shape in tf.keras
    Dropout = get_dropout()

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio

    if block_args.expand_ratio != 1:

        x = tf.keras.layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + 'expand_bn')(x)
        x = tf.keras.layers.Activation(activation, name=prefix + 'expand_activation')(x)

    else:
        x = inputs

    # Depthwise Convolution
    x = tf.keras.layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dwconv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + 'bn')(x)
    x = tf.keras.layers.Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = tf.keras.layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) if tf.keras.backend.image_data_format() == 'channels_last' else (filters, 1, 1)

        se_tensor = tf.keras.layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)

        se_tensor = tf.keras.layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)

        se_tensor = tf.keras.layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)
 
        x = tf.keras.layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = tf.keras.layers.Conv2D(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)

    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + 'project_bn')(x)
    
    if block_args.id_skip and all(s == 1 for s in block_args.strides) \
                    and block_args.input_filters == block_args.output_filters:
     
        if drop_rate and (drop_rate > 0):
            x = Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=prefix + 'drop')(x)
        
        x = tf.keras.layers.add([x, inputs], name=prefix + 'add')

    return x




def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None, # (128, 256, 12).
                 pooling=None,
                 classes=1000,
                 **kwargs # no.
                 ):
  
    img_input = tf.keras.Input(shape=input_shape)    
    # img = tf.keras.layers.Permute((2, 3, 1))(img_input) # (b, 128, 256, 12).
    # img = img_input

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    activation = get_swish()

    # Build stem
    x = img_input
    x = tf.keras.layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = tf.keras.layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    num_blocks_total = sum(round_repeats(block_args.num_repeat,
                                         depth_coefficient) for block_args in blocks_args)


    block_num = 0

    for idx, block_args in enumerate(blocks_args):

        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters, width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))
 
        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))


        block_num += 1
        if block_args.num_repeat > 1:

            # pylint: disable=protected-access
            block_args = block_args._replace(\
                        input_filters=block_args.output_filters, strides=[1, 1])

            # pylint: enable=protected-access
            # for bidx in xrange(block_args.num_repeat - 1):
            for bidx in range(block_args.num_repeat - 1):

                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(idx + 1, string.ascii_lowercase[bidx + 1])
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

 
    # Build top
    x = tf.keras.layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = tf.keras.layers.Activation(activation, name='top_activation')(x)



    x = tf.keras.layers.BatchNormalization(axis=3)(x) # 4, 8, 1408
    x = tf.keras.layers.Conv2D(32, 1, padding='valid')(x)  # (b, 4, 8, 32).
    x = tf.keras.layers.BatchNormalization(axis=3)(x) # (b, 4, 8, 32).
    x = tf.keras.layers.Flatten()(x) # (b, 1024).
    x = tf.keras.layers.ELU()(x)
    '''
        in shape: (b, 128, 256, 12) .
        out shape: (b, 1024). 
    '''
    model = tf.keras.Model(img_input, x, name=model_name)
 
    return model


def EfficientNetB2(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=(128, 256, 12).,
                pooling=None,
                classes=1000,
                **kwargs):

    '''
        in shape: (b, 128, 256, 3) .
        out shape: (b, 4, 8, 1408).
        ch format: last.
    '''
    return EfficientNet(
        1.1, 1.2, 260, 0.3,
        model_name='efficientnet-b2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )



 
if __name__ == '__main__': 

    model = EfficientNetB2() 
    model(tf.random.uniform((2,128,256,6))).shape
