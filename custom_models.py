# Author: Lichirui Zhang, Columbia University
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Activation, Dropout, Conv1D, MaxPool1D, BatchNormalization
from tensorflow_addons.layers import InstanceNormalization
import math
import numpy as np
from tensorflow.python.keras import backend as K

################# ResNet ###############
class ResnetLayer1D(tf.keras.layers.Layer):
    def __init__(self, d_out=128, kernel_size=3, strides=1, dropout=0.15):
        super(ResnetLayer1D, self).__init__() 
        self.conv0 = Conv1D(d_out, kernel_size, strides=strides, padding='same', use_bias=False)
        self.norm0 = BatchNormalization(momentum=0.9265)
        # trunk
        self.conv1 = Conv1D(d_out, 1, strides=strides, padding='same', use_bias=False)
        self.norm1 = BatchNormalization(momentum=0.9265)
        self.conv2 = Conv1D(d_out, kernel_size, padding='same', use_bias=False)
        self.norm2 = BatchNormalization(momentum=0.9265)
        self.dropout = Dropout(dropout)
        self.conv3 = Conv1D(d_out, 1, padding='same', use_bias=False)
               
    def call(self, x_in):
        
        # main trunk
        x = self.norm1(x_in)
        x = Activation('elu')(x)
        x = self.conv1(x) # c//2
        x = self.dropout(x)
        x = self.norm2(x)
        x = Activation('elu')(x)
        x = self.conv2(x)
        
        x_in = self.norm0(x_in)
        x_in = Activation('elu')(x_in)
        x_in = self.conv0(x_in)
        x_out = x + x_in
        return x_out

class ResnetLayer1D_downsample(tf.keras.layers.Layer):
    def __init__(self, d_out=128, kernel_size=5, strides=2, dropout=0.15):
        super(ResnetLayer1D_downsample, self).__init__()            
        self.conv0 = Conv1D(d_out, kernel_size, strides=strides, padding='same', use_bias=False)
        self.norm0 = BatchNormalization(momentum=0.9265)
        # trunk
        self.conv1 = Conv1D(d_out, 1, strides=strides, padding='same', use_bias=False)
        self.norm1 = BatchNormalization(momentum=0.9265)
        self.conv2 = Conv1D(d_out, kernel_size, padding='same', use_bias=False)
        self.norm2 = BatchNormalization(momentum=0.9265)
        self.dropout = Dropout(dropout)
        self.conv3 = Conv1D(d_out, 1, padding='same', use_bias=False)
               
    def call(self, x_in):
        # main trunk
        x = self.norm1(x_in)
        x = Activation('elu')(x)
        x = self.conv1(x) # c//2
        x = self.dropout(x)
        x = self.norm2(x)
        x = Activation('elu')(x)
        x = self.conv2(x)
        
        
        x_in = self.norm0(x_in)
        x_in = Activation('elu')(x_in)
        x_in = self.conv0(x_in)
        x_out = x + x_in
        return x_out

class Residual_downsample_1D(tf.keras.layers.Layer):
    def __init__(self, d_out=128, kernel_size=5, strides=1, dropout=0.15):
        super(Residual_downsample_1D, self).__init__()            
        # trunk
        self.conv1 = Conv1D(d_out, kernel_size, strides=strides, padding='same', use_bias=False)
        self.norm1 = BatchNormalization(momentum=0.9265)
        self.dropout = Dropout(dropout)
    def call(self, x_in):
        # main trunk
        x = Activation('relu')(x_in)
        x = self.conv1(x) # c//2
        x = self.norm1(x)
        x = self.dropout(x)
        x += x_in
        x = MaxPool1D(2, padding='same')(x)
        return x
    
class Residual_dilated_1D(tf.keras.layers.Layer):
    def __init__(self, d_out=96, kernel_size=3, strides=1, dilation_rate=1, dropout=0.15):
        super(Residual_dilated_1D, self).__init__()            
        # trunk
        self.conv1 = Conv1D(d_out//2, kernel_size, dilation_rate=dilation_rate, strides=strides, padding='same', use_bias=False)
        self.norm1 = BatchNormalization(momentum=0.9265)
        self.conv2 = Conv1D(d_out, 1, strides=strides, padding='same', use_bias=False)
        self.norm2 = BatchNormalization(momentum=0.9265)
        self.dropout = Dropout(dropout)
        
    def call(self, x_in):
        # main trunk
        x = Activation('relu')(x_in)
        x = self.conv1(x) # c//2
        x = self.norm1(x)
        
        x = Activation('relu')(x)
        x = self.conv2(x) 
        x = self.norm2(x)
        x = self.dropout(x)
        x += x_in
        return x

class Residual_dilated_2D(tf.keras.layers.Layer):
    def __init__(self, d_out=48, kernel_size=3, strides=1, dilation_rate=4, dropout=0.15):
        super(Residual_dilated_2D, self).__init__()            
        # trunk
        self.conv1 = Conv2D(d_out//2, kernel_size, dilation_rate=dilation_rate, strides=strides, padding='same', use_bias=False)
        self.norm1 = BatchNormalization(momentum=0.9265)
        self.conv2 = Conv2D(d_out, 1, strides=strides, padding='same', use_bias=False)
        self.norm2 = BatchNormalization(momentum=0.9265)
        self.dropout = Dropout(dropout)
        
    def call(self, x_in):
        # main trunk
        x = Activation('relu')(x_in)
        x = self.conv1(x) # c//2
        x = self.norm1(x)
        
        x = Activation('relu')(x)
        x = self.conv2(x) 
        x = self.norm2(x)
        x = self.dropout(x)
        x += x_in
        return x


class ResnetLayer2D(tf.keras.layers.Layer):
    def __init__(self, d_out=128, kernel_size=3, strides=1, dropout=0.15):
        super(ResnetLayer2D, self).__init__()  
        
        # trunk
        self.conv1 = Conv2D(d_out, 1, strides=strides, padding='same', use_bias=False)
        self.norm1 = BatchNormalization(momentum=0.9265)
        self.conv2 = Conv2D(d_out, kernel_size, padding='same', use_bias=False)
        self.norm2 = BatchNormalization(momentum=0.9265)
        self.dropout = Dropout(dropout)
        self.conv3 = Conv2D(d_out, 1, padding='same', use_bias=False)
               
    def call(self, x_in):
        
        # main trunk
        x = self.norm1(x_in)
        x = Activation('elu')(x)
        x = self.conv1(x) # c//2
        x = self.dropout(x)
        x = self.norm2(x)
        x = Activation('elu')(x)
        x = self.conv2(x)
        x_out = x + x_in
        return x_out

class ResnetLayer2D_downsample(tf.keras.layers.Layer):
    def __init__(self, d_out=128, kernel_size=5, strides=2, dropout=0.15):
        super(ResnetLayer2D_downsample, self).__init__()            
        self.conv0 = Conv2D(d_out, kernel_size, strides=strides, padding='same', use_bias=False)
        self.norm0 = BatchNormalization(momentum=0.9265)
        # trunk
        conv1 = Conv2D(d_out, 1, strides=strides, padding='same', use_bias=False)
        norm1 = BatchNormalization(momentum=0.9265)
        conv2 = Conv2D(d_out, kernel_size, padding='same', use_bias=False)
        norm2 = BatchNormalization(momentum=0.9265)
        dropout = Dropout(dropout)
        conv3 = Conv2D(d_out, 1, padding='same', use_bias=False)
               
    def call(self, x_in):
        # main trunk
        x = self.norm1(x_in)
        x = Activation('elu')(x)
        x = self.conv1(x) # c//2
        x = self.dropout(x)
        x = self.norm2(x)
        x = Activation('elu')(x)
        x = self.conv2(x)
        
        
        x_in = self.norm0(x_in)
        x_in = Activation('elu')(x_in)
        x_in = self.conv0(x_in)
        x_out = x + x_in
        return x_out
    
############## Transformers #################
def point_wise_feed_forward_network(d_model, dff, activation='relu', rate=0.1):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=activation),  # (batch_size, seq_len, dff)
      Dropout(rate),
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        self.dropout = Dropout(dropout)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        x = tf.transpose(x, perm=[0,2,1,3])
        return x

    def call(self, q, k=None, v=None, attention_mask=None):
        batch_size = tf.shape(q)[0]
        if k is None and v is None:
            k = q
            v = q

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if attention_mask is not None:
            scaled_attention_logits += (attention_mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        attention_weights = self.dropout(attention_weights)

        scaled_attention = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output

# pre-layernorm
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = tf.keras.layers.LayerNormalization()
#         self.layernorm2 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def build(self, input_shape):
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

    def call(self, x, mask=None):
        
        attn_output = self.mha(self.layernorm1(x), attention_mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = x + self.dropout1(attn_output)

        ffn_output = attn_output + self.dropout2(self.ffn(attn_output)) 
        return ffn_output
    
#         attn_output = x + self.mha(x, attention_mask=mask)  # (batch_size, input_seq_len, d_model)
#         attn_output = self.layernorm1(attn_output)
#         attn_output = self.dropout1(attn_output)
#         ffn_output = attn_output + self.ffn(attn_output)  # (batch_size, input_seq_len, d_model)
#         ffn_output = self.layernorm2(ffn_output)
#         return ffn_output




# Axial transformer encoder layer. It takes a 4-D input (B, N, L, d), and update it with the axial attention mechanism.
class AxialEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(AxialEncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.mha_1 = MultiHeadAttention(d_model, num_heads)
        self.mha_2 = MultiHeadAttention(d_model, num_heads)

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.layernorm3 = tf.keras.layers.LayerNormalization()
        self.layernorm4 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)
    def build(self, input_shape):
        self.ffn1 = point_wise_feed_forward_network(self.d_model, self.dff, rate=self.rate)
        self.ffn2 = point_wise_feed_forward_network(self.d_model, self.dff, rate=self.rate)

    def call(self, x_in, mask=None):
        B = tf.shape(x_in)[0]
        N = tf.shape(x_in)[1]
        L = tf.shape(x_in)[2]
        
        # attention over L
        x = self.layernorm1(x_in)
        x = tf.reshape(x, (B*N, L, self.d_model))
        x = self.mha_1(x, attention_mask=mask)
        x = tf.reshape(x, (B, N, L, self.d_model))
        x_in = x_in + self.dropout1(x)
        
        # ffn
        x = self.layernorm2(x_in)
        x = self.ffn1(x)
        x_in = x_in + self.dropout2(x)
        
        # attention over N
        x = tf.transpose(x_in, [0, 2, 1, 3])
        x = self.layernorm3(x)
        x = tf.reshape(x, (B*L, N, self.d_model))
        x = self.mha_2(x, attention_mask=mask)
        x = tf.reshape(x, (B, L, N, self.d_model))
        x_in = x_in + self.dropout3(x)
        
        # ffn
        x = self.layernorm4(x_in)
        x = self.ffn2(x)
        x_in = x_in + self.dropout4(x)
        
        x_in  = tf.transpose(x_in, [0, 2, 1, 3])
        return x_in

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def positionalencoding2d(height, width, d_model):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = np.zeros((d_model, height, width))
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = np.exp(np.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = np.expand_dims(np.arange(0., width), 1) # (W, D/4)
    pos_h = np.expand_dims(np.arange(0., height), 1)
    print(pe.shape, np.sin(pos_w * div_term).shape)
    pe[0:d_model:2, :, :] = np.expand_dims(np.sin(pos_w * div_term).transpose(1, 0), 1).repeat(height, 1) # ()
    pe[1:d_model:2, :, :] = np.expand_dims(np.cos(pos_w * div_term).transpose(1, 0), 1).repeat(height, 1)
    pe[d_model::2, :, :] = np.expand_dims(np.sin(pos_h * div_term).transpose(1, 0), 2).repeat(width, 2)
    pe[d_model + 1::2, :, :] = np.expand_dims(np.cos(pos_h * div_term).transpose(1, 0), 2).repeat(width, 2)

    return np.transpose(pe, (1,2,0))[np.newaxis, ...] # (B, H, W, D)



############ helpers ############
class Symmetrize(tf.keras.layers.Layer):
    def __init__(self):
        super(Symmetrize, self).__init__()
    def call(self, x):
        return (x + tf.transpose(x, [0, 2, 1, 3]))/2





# class EncoderLayer(tf.keras.layers.Layer):
#     def __init__(self, d_model, num_heads, dff, rate=0.1):
#         super(EncoderLayer, self).__init__()
#         self.d_model = d_model
#         self.dff = dff
#         self.mha = MultiHeadAttention(d_model, num_heads)
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = tf.keras.layers.Dropout(rate)
        
#     def build(self, input_shape):
#         self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

#     def call(self, x, mask=None):
#         attn_output = x + self.mha(x, attention_mask=mask)  # (batch_size, input_seq_len, d_model)
#         attn_output = self.layernorm1(attn_output)
#         attn_output = self.dropout1(attn_output)
#         ffn_output = attn_output + self.ffn(attn_output)  # (batch_size, input_seq_len, d_model)
#         ffn_output = self.layernorm2(ffn_output)
#         return ffn_output
    
class UpperTri(tf.keras.layers.Layer):
    def __init__(self, diagonal_offset=2):
        super(UpperTri, self).__init__()
        self.diagonal_offset = diagonal_offset

    def call(self, inputs):
        seq_len = inputs.shape[1]
        output_dim = inputs.shape[-1]

        if type(seq_len) == tf.compat.v1.Dimension:
            seq_len = seq_len.value
            output_dim = output_dim.value

        triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
        triu_index = list(triu_tup[0]+ seq_len*triu_tup[1])
        unroll_repr = tf.reshape(inputs, [-1, seq_len**2, output_dim])
        return tf.gather(unroll_repr, triu_index, axis=1)




class StochasticReverseComplement(tf.keras.layers.Layer):
    def __init__(self):
        super(StochasticReverseComplement, self).__init__()
    def call(self, seq_1hot, training=None):
        if training:
            rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
            rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
            reverse_bool = tf.random.uniform(shape=[]) > 0.5
            src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, tf.constant(False)
def shift_sequence(seq, shift_amount, pad_value=0.25):
  if seq.shape.ndims != 3:
    raise ValueError('input sequence should be rank 3')
  input_shape = seq.shape

  pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift_amount), :])

  def _shift_right(_seq):
    sliced_seq = _seq[:, :-shift_amount:, :]
    return tf.concat([pad, sliced_seq], axis=1)

  def _shift_left(_seq):
    sliced_seq = _seq[:, -shift_amount:, :]
    return tf.concat([sliced_seq, pad], axis=1)

  output = tf.cond(
      tf.greater(shift_amount, 0), lambda: _shift_right(seq),
      lambda: _shift_left(seq))

  output.set_shape(input_shape)
  return output

class StochasticShift(tf.keras.layers.Layer):
    def __init__(self, shift_max=0, symmetric=True, pad='uniform'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = tf.range(-self.shift_max, self.shift_max+1)
        else:
            self.augment_shifts = tf.range(0, self.shift_max+1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(shape=[], minval=0, dtype=tf.int64,
                                  maxval=len(self.augment_shifts))
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                              lambda: shift_sequence(seq_1hot, shift),
                              lambda: seq_1hot)
            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({
          'shift_max': self.shift_max,
          'symmetric': self.symmetric,
          'pad': self.pad
        })
        return config

# class SwitchReverse(tf.keras.layers.Layer):
#     def __init__(self, strand_pair=None):
#         super(SwitchReverse, self).__init__()
#         self.strand_pair = strand_pair
#     def call(self, x_reverse):
#         x = x_reverse[0]
#         reverse = x_reverse[1]

#         xd = len(x.shape)
#         if xd == 3:
#             rev_axes = [1]
#         elif xd == 4:
#             rev_axes = [1,2]
#         else:
#             raise ValueError('Cannot recognize SwitchReverse input dimensions %d.' % xd)

#         xr = tf.keras.backend.switch(reverse,
#                                      tf.reverse(x, axis=rev_axes),
#                                      x)

#         if self.strand_pair is None:
#             xrs = xr
#         else:
#             xrs = tf.keras.backend.switch(reverse,
#                                         tf.gather(xr, self.strand_pair, axis=-1),
#                                         xr)
    
#         return xrs

class SwitchReverseTriu(tf.keras.layers.Layer):
    def __init__(self, diagonal_offset):
        super(SwitchReverseTriu, self).__init__()
        self.diagonal_offset = diagonal_offset

    def call(self, x_reverse):
        x_ut = x_reverse[0]
        reverse = x_reverse[1]

      # infer original sequence length
        ut_len = x_ut.shape[1]
        if type(ut_len) == tf.compat.v1.Dimension:
            ut_len = ut_len.value
        seq_len = int(np.sqrt(2*ut_len + 0.25) - 0.5)
        seq_len += self.diagonal_offset

        # get triu indexes
        ut_indexes = np.triu_indices(seq_len, self.diagonal_offset)
        assert(len(ut_indexes[0]) == ut_len)

        # construct a ut matrix of ut indexes
        mat_ut_indexes = np.zeros(shape=(seq_len,seq_len), dtype='int')
        mat_ut_indexes[ut_indexes] = np.arange(ut_len)

        # make lower diag mask
        mask_ut = np.zeros(shape=(seq_len,seq_len), dtype='bool')
        mask_ut[ut_indexes] = True
        mask_ld = ~mask_ut

        # construct a matrix of symmetric ut indexes
        mat_indexes = mat_ut_indexes + np.multiply(mask_ld, mat_ut_indexes.T)

        # reverse complement
        mat_rc_indexes = mat_indexes[::-1,::-1]

        # extract ut order
        rc_ut_order = mat_rc_indexes[ut_indexes]

        return tf.keras.backend.switch(reverse,
                                       tf.gather(x_ut, rc_ut_order, axis=1),
                                       x_ut)
    def get_config(self):
        config = super().get_config().copy()
        config['diagonal_offset'] = self.diagonal_offset
        return config
    
class OneToTwo(tf.keras.layers.Layer):
  def __init__(self, operation='mean'):
    super(OneToTwo, self).__init__()
    self.operation = operation.lower()
    valid_operations = ['concat','mean','max','multipy','multiply1']
    assert self.operation in valid_operations

  def call(self, oned):
    _, seq_len, features = oned.shape

    twod1 = tf.tile(oned, [1, seq_len, 1])
    twod1 = tf.reshape(twod1, [-1, seq_len, seq_len, features])
    twod2 = tf.transpose(twod1, [0,2,1,3])

    if self.operation == 'concat':
      twod  = tf.concat([twod1, twod2], axis=-1)

    elif self.operation == 'multiply':
      twod  = tf.multiply(twod1, twod2)

    elif self.operation == 'multiply1':
      twod = tf.multiply(twod1+1, twod2+1) - 1

    else:
      twod1 = tf.expand_dims(twod1, axis=-1)
      twod2 = tf.expand_dims(twod2, axis=-1)
      twod  = tf.concat([twod1, twod2], axis=-1)

      if self.operation == 'mean':
        twod = tf.reduce_mean(twod, axis=-1)

      elif self.operation == 'max':
        twod = tf.reduce_max(twod, axis=-1)

    return twod
################################################################################
# Metrics
################################################################################
class SeqAUC(tf.keras.metrics.AUC):
  def __init__(self, curve='ROC', name=None, summarize=True, **kwargs):
    if name is None:
      if curve == 'ROC':
        name = 'auroc'
      elif curve == 'PR':
        name = 'auprc'
    super(SeqAUC, self).__init__(curve=curve, name=name, multi_label=True, **kwargs)
    self._summarize = summarize
    

  def update_state(self, y_true, y_pred, **kwargs):
    """Flatten sequence length before update."""

    # flatten batch and sequence length
    num_targets = y_pred.shape[-1]
    y_true = tf.reshape(y_true, (-1,num_targets))
    y_pred = tf.reshape(y_pred, (-1,num_targets))

    # update
    super(SeqAUC, self).update_state(y_true, y_pred, **kwargs)


  def interpolate_pr_auc(self):
    """Add option to remove summary."""
    dtp = self.true_positives[:self.num_thresholds -
                              1] - self.true_positives[1:]
    p = tf.math.add(self.true_positives, self.false_positives)
    dp = p[:self.num_thresholds - 1] - p[1:]
    prec_slope = tf.math.divide_no_nan(
        dtp, tf.maximum(dp, 0), name='prec_slope')
    intercept = self.true_positives[1:] - tf.multiply(prec_slope, p[1:])

    safe_p_ratio = tf.where(
        tf.logical_and(p[:self.num_thresholds - 1] > 0, p[1:] > 0),
        tf.math.divide_no_nan(
            p[:self.num_thresholds - 1],
            tf.maximum(p[1:], 0),
            name='recall_relative_ratio'),
        tf.ones_like(p[1:]))

    pr_auc_increment = tf.math.divide_no_nan(
        prec_slope * (dtp + intercept * tf.math.log(safe_p_ratio)),
        tf.maximum(self.true_positives[1:] + self.false_negatives[1:], 0),
        name='pr_auc_increment')

    if self.multi_label:
      by_label_auc = tf.reduce_sum(
          pr_auc_increment, name=self.name + '_by_label', axis=0)

      if self._summarize:
        if self.label_weights is None:
          # Evenly weighted average of the label AUCs.
          return tf.reduce_mean(by_label_auc, name=self.name)
        else:
          # Weighted average of the label AUCs.
          return tf.math.divide_no_nan(
              tf.reduce_sum(
                  tf.multiply(by_label_auc, self.label_weights)),
              tf.reduce_sum(self.label_weights),
              name=self.name)
      else:
        return by_label_auc
    else:
      if self._summarize:
        return tf.reduce_sum(pr_auc_increment, name='interpolate_pr_auc')
      else:
        return pr_auc_increment


  def result(self):
    """Add option to remove summary.
    It's not clear why, but these metrics_utils == aren't working for tf.26 on.
    I'm hacking a solution to compare the values instead."""
    if (self.curve.value == metrics_utils.AUCCurve.PR.value and
        self.summation_method.value == metrics_utils.AUCSummationMethod.INTERPOLATION.value
       ):
      # This use case is different and is handled separately.
      return self.interpolate_pr_auc()

    # Set `x` and `y` values for the curves based on `curve` config.
    recall = tf.math.divide_no_nan(
        self.true_positives,
        tf.math.add(self.true_positives, self.false_negatives))
    if self.curve.value == metrics_utils.AUCCurve.ROC.value:
      fp_rate = tf.math.divide_no_nan(
          self.false_positives,
          tf.math.add(self.false_positives, self.true_negatives))
      x = fp_rate
      y = recall
    else:  # curve == 'PR'.
      precision = tf.math.divide_no_nan(
          self.true_positives,
          tf.math.add(self.true_positives, self.false_positives))
      x = recall
      y = precision

    # Find the rectangle heights based on `summation_method`.
    if self.summation_method.value == metrics_utils.AUCSummationMethod.INTERPOLATION.value:
      # Note: the case ('PR', 'interpolation') has been handled above.
      heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.
    elif self.summation_method.value == metrics_utils.AUCSummationMethod.MINORING.value:
      heights = tf.minimum(y[:self.num_thresholds - 1], y[1:])
    else:  # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
      heights = tf.maximum(y[:self.num_thresholds - 1], y[1:])

    # Sum up the areas of all the rectangles.
    if self.multi_label:
      riemann_terms = tf.multiply(x[:self.num_thresholds - 1] - x[1:],
                                        heights)
      by_label_auc = tf.reduce_sum(
          riemann_terms, name=self.name + '_by_label', axis=0)

      if self._summarize:
        if self.label_weights is None:
          # Unweighted average of the label AUCs.
          return tf.reduce_mean(by_label_auc, name=self.name)
        else:
          # Weighted average of the label AUCs.
          return tf.math.div_no_nan(
              tf.reduce_sum(
                  tf.multiply(by_label_auc, self.label_weights)),
              tf.reduce_sum(self.label_weights),
              name=self.name)
      else:
        return by_label_auc
    else:
      if self._summarize:
        return tf.reduce_sum(
            tf.multiply(x[:self.num_thresholds-1] - x[1:], heights),
            name=self.name)
      else:
        return tf.multiply(x[:self.num_thresholds-1] - x[1:], heights)


class PearsonR(tf.keras.metrics.Metric):
  def __init__(self, num_targets, summarize=True, name='pearsonr', **kwargs):
    super(PearsonR, self).__init__(name=name, **kwargs)
    self._summarize = summarize
    self._shape = (num_targets,)
    self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

    self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
    self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
    self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')
    self._pred_sum = self.add_weight(name='pred_sum', shape=self._shape, initializer='zeros')
    self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    if len(y_true.shape) == 2:
      reduce_axes = 0
    else:
      reduce_axes = [0,1]

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
    self._product.assign_add(product)

    true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
    self._true_sumsq.assign_add(true_sumsq)

    pred_sum = tf.reduce_sum(y_pred, axis=reduce_axes)
    self._pred_sum.assign_add(pred_sum)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
    self._pred_sumsq.assign_add(pred_sumsq)

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=reduce_axes)
    self._count.assign_add(count)

  def result(self):
    true_mean = tf.divide(self._true_sum, self._count)
    true_mean2 = tf.math.square(true_mean)
    pred_mean = tf.divide(self._pred_sum, self._count)
    pred_mean2 = tf.math.square(pred_mean)

    term1 = self._product
    term2 = -tf.multiply(true_mean, self._pred_sum)
    term3 = -tf.multiply(pred_mean, self._true_sum)
    term4 = tf.multiply(self._count, tf.multiply(true_mean, pred_mean))
    covariance = term1 + term2 + term3 + term4

    true_var = self._true_sumsq - tf.multiply(self._count, true_mean2)
    pred_var = self._pred_sumsq - tf.multiply(self._count, pred_mean2)
    pred_var = tf.where(tf.greater(pred_var, 1e-12),
                        pred_var,
                        np.inf*tf.ones_like(pred_var))
    
    tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
    correlation = tf.divide(covariance, tp_var)

    if self._summarize:
        return tf.reduce_mean(correlation)
    else:
        return correlation

  def reset_states(self):
      K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])


class R2(tf.keras.metrics.Metric):
  def __init__(self, num_targets, summarize=True, name='r2', **kwargs):
    super(R2, self).__init__(name=name, **kwargs)
    self._summarize = summarize
    self._shape = (num_targets,)
    self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

    self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
    self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')

    self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
    self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    if len(y_true.shape) == 2:
      reduce_axes = 0
    else:
      reduce_axes = [0,1]

    true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
    self._true_sumsq.assign_add(true_sumsq)

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
    self._product.assign_add(product)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
    self._pred_sumsq.assign_add(pred_sumsq)

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=reduce_axes)
    self._count.assign_add(count)

  def result(self):
    true_mean = tf.divide(self._true_sum, self._count)
    true_mean2 = tf.math.square(true_mean)

    total = self._true_sumsq - tf.multiply(self._count, true_mean2)

    resid1 = self._pred_sumsq
    resid2 = -2*self._product
    resid3 = self._true_sumsq
    resid = resid1 + resid2 + resid3

    r2 = tf.ones_like(self._shape, dtype=tf.float32) - tf.divide(resid, total)

    if self._summarize:
        return tf.reduce_mean(r2)
    else:
        return r2

  def reset_states(self):
    K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])

import scipy.stats as stats
def spearmanr(targets, preds, num_targets=5):

    scor = np.zeros(5)

    for ti in range(num_targets):
#   if self.targets_na is not None:
#     preds_ti = self.preds[~self.targets_na, ti]
#     targets_ti = self.targets[~self.targets_na, ti]
#   else:
        preds_ti = preds[:, :, ti].flatten()
        targets_ti = targets[:, :, ti].flatten()

    sc, _ = stats.spearmanr(targets_ti, preds_ti)
    scor[ti] = sc

    return scor
