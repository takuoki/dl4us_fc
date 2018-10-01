# 00_util

import numpy as np

# detoken
def detoken(detokenizer, seqs):
    results = []
    for seq in seqs:
        detokened_seq = []
        for i in seq:
            if i == 0:
                break
            detokened_seq.append(detokenizer[i])
        results.append(' '.join(detokened_seq))
    return results

# translate
def translate(detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len, test_count=5):
    en_seqs = detoken(detokenizer_en, x_valid[:test_count])
    ja_pred_seqs = detoken(detokenizer_ja, predict_all(generator_model, x_valid[:test_count], ja_seq_len))
    ja_seqs = detoken(detokenizer_ja, y_valid[:test_count])

    for i in range(test_count):
        print('EN(', i, '): ', en_seqs[i])
        print('JA-predict(', i, '): ', ja_pred_seqs[i])
        print('JA-answer (', i, '): ', ja_seqs[i])

# onehot
#   (n, seq_len) -> (n, seq_len, vocab_size)
def onehot(seqs, vocab_size):
    return np.identity(vocab_size)[seqs]

# TODO: 精度検証(BLUE?)

# MyEmbedding
# one hot表現をインプットにするEmbededdingレイヤー
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer
from keras.legacy import interfaces

class MyEmbedding(Layer):

    @interfaces.legacy_embedding_support
    def __init__(self, input_dim, output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(MyEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        output_mask = K.not_equal(inputs, 0)
        return output_mask

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.output_dim,)

    def call(self, inputs):
        inputs = K.cast(inputs, 'float32')
        out = K.dot(inputs[1:], self.embeddings)
        return out

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'embeddings_initializer':
                      initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer':
                      regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint':
                      constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero,
                  'input_length': self.input_length}
        base_config = super(Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))