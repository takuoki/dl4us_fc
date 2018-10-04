# 03_discriminator

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

# discriminator
#   model : (en_seq_len, ja_seq_len) -> (2) : True/False
def discriminator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size, emb_dim=256, hid_dim=256, opt=Adam(lr=1e-4)):

    encoder_inputs, _, encoder_states = encoder(en_seq_len, en_vocab_size, emb_dim, hid_dim)
    decoder_inputs, decoder_outputs = decoder(encoder_states, ja_seq_len, ja_vocab_size, emb_dim, hid_dim)

    x = Flatten()(decoder_outputs)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    model_output = Dense(2, activation='softmax')(x)

    model = Model([encoder_inputs, decoder_inputs], model_output)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model, encoder_inputs
