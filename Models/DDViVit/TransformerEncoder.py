
import math

import six
import tensorflow as tf
  
import numpy as np

from parameter import transformerEncoder_args as args

from einops.layers.tensorflow import Rearrange


class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]

 

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(identifier):

    if isinstance(identifier, six.string_types):
        name_to_fn = {"gelu": gelu}
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)



class KVQ(tf.keras.layers.Layer):

    def __init__(self, **kwarg):
        super(KVQ, self).__init__(**kwarg)

    def build(self, input_shape):

        self.kernel = self.add_weight("kernel",
                                    shape=[args.hidden_size, args.heads, args.hidden_size//args.heads],
                                    initializer=tf.keras.initializers.RandomNormal(),
                                    dtype=tf.float32)
        self.bias = self.add_weight("bias",
                                     shape=[args.heads, args.hidden_size//args.heads],
                                     initializer=tf.keras.initializers.RandomNormal(),
                                     dtype=tf.float32)
        super(KVQ, self).build(input_shape)


    def call(self, x): # x.shape: (64, 65, 64) = (64, 65, hidden_size)
        # (512, 50, 768) * (768, 12, 64) -> (512, 50, 12, 64)
        return tf.einsum('bij,jhd->bihd', x, self.kernel) + self.bias



class AttentionOut(tf.keras.layers.Layer):

    def __init__(self, **kwarg):
        super(AttentionOut, self).__init__(**kwarg)

    def build(self, input_shape):

        self.kernel = self.add_weight("kernel",
                                             shape=[args.heads, args.hidden_size//args.heads, args.hidden_size],
                                             initializer=tf.keras.initializers.RandomNormal(),
                                             dtype=tf.float32)
        self.bias = self.add_weight("bias",
                                             shape=[args.hidden_size],
                                             initializer=tf.keras.initializers.RandomNormal(),
                                             dtype=tf.float32)
        super(AttentionOut, self).build(input_shape)

    def call(self, x):
        # (b p h d), (h d hid) -> (b p hid)
        return tf.einsum('bphd,hdk->bpk', x, self.kernel) + self.bias





class Attention(tf.keras.layers.Layer):

    def __init__(self, **kwarg):
        super(Attention, self).__init__(**kwarg)

    def build(self, input_shape):

        self.scale = args.hidden_size ** -0.5

        self.rearrange_qkv = Rearrange('b p h d -> b h p d')

        self.rearrange_out = Rearrange('b h p d -> b p h d')

        self.K = KVQ(name = 'key')
        self.V = KVQ(name = 'value')
        self.Q = KVQ(name = 'query')
        self.out = AttentionOut(name = 'out')

        super(Attention, self).build(input_shape)



    def call(self, x): # x.shape: (64, 65, 64) = (64, 65, hidden_size)

        k = self.K(x)
        v = self.V(x)
        q = self.Q(x)

        # (b h p d)
        k = self.rearrange_qkv(k)
        v = self.rearrange_qkv(v)
        q = self.rearrange_qkv(q)

        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale #  A = K_t * Q
        attn = tf.nn.softmax(dots, axis=-1)

        # (b h p d)
        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        # (b p h d)
        out = self.rearrange_out(out)


        return self.out(out)



class MlpBlock(tf.keras.layers.Layer):

    def __init__(self, **kwarg):
        super(MlpBlock, self).__init__(**kwarg)

    def build(self, input_shape):
        self.seq = []
        self.seq.append(tf.keras.layers.Dense(args.mlp_dim, activation=get_activation('gelu'), name='Dense_0'))
        self.seq.append(tf.keras.layers.Dropout(args.dropout_rate))
        self.seq.append(tf.keras.layers.Dense(args.hidden_size, name='Dense_1'))
        self.seq.append(tf.keras.layers.Dropout(args.dropout_rate))

        super(MlpBlock, self).build(input_shape)

    def call(self, x, training=False): # x.shape: (64, 65, 64) = (64, 65, hidden_size)

        for layer in self.seq: x = layer(x, training=training)
        return x




class Encoder1DBlock(tf.keras.layers.Layer):

    def __init__(self, **kwarg):
        super(Encoder1DBlock, self).__init__(**kwarg)

    def build(self, input_shape):

        self.seq1 = []
        self.seq1.append(tf.keras.layers.LayerNormalization(name='LayerNorm_0', epsilon=1e-5))
        self.seq1.append(Attention(name='MultiHeadDotProductAttention_1'))
        self.seq1.append(tf.keras.layers.Dropout(args.dropout_rate))

        self.seq2 = []
        self.seq2.append(tf.keras.layers.LayerNormalization(name='LayerNorm_2', epsilon=1e-5))
        self.seq2.append(MlpBlock(name='MlpBlock_3'))

        super(Encoder1DBlock, self).build(input_shape)

    def call(self, x, training=False): # x.shape: (64, 65, 64) = (64, 65, hidden_size)

        x1 = x
        for layer in self.seq1: x = layer(x, training=training)
        x = x + x1
        x1 = x
        for layer in self.seq2: x = layer(x, training=training)
        return x + x1

 

class TransformerEncoder(tf.keras.layers.Layer): # Transformer's encoder part only.
 
    def __init__(self, **kwarg):
        super(TransformerEncoder, self).__init__(**kwarg)


    def build(self, input_shape):

        self.seq = []

        self.seq.append(tf.keras.layers.Dropout(args.dropout_rate))
        for i in range(args.depth):
            self.seq.append(Encoder1DBlock(name=f'encoderblock_{i}'))

        self.seq.append(tf.keras.layers.LayerNormalization(epsilon=1e-5, name='encoder_norm'))


    def call(self, x, training=False):
        # return self.model(x)
        for layer in self.seq: x = layer(x, training=training)
        return x
