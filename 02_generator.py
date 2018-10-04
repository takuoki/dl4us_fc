# 02_generator

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.layers.core import Dense

# encoder
#   input_layer : input layer for en sentense (size=seq_len)
#   outputs     : encoded sentense (size=(seq_len, hid_dim))
#   states      : hidden states (size=???)
def encoder(seq_len, vocab_size, emb_dim=256, hid_dim=256):

    # layer
    input_layer = Input(shape=(seq_len,))
    embededding_layer = Embedding(vocab_size, emb_dim, mask_zero=True)
    lstm_layer = LSTM(hid_dim, return_sequences=True, return_state=True)

    # connect layer
    # TODO: LSTMレイヤの多層化、双方向化
    outputs, *states = lstm_layer(embededding_layer(input_layer))

    return input_layer, outputs, states

# decoder
#   input_layer : input layer for one ja character (size=(seq_len,))
#   outputs     : encoded sentense (size=(seq_len, hid_dim))
def decoder(encoder_states, seq_len, vocab_size, emb_dim=256, hid_dim=256):

    # layer
    input_layer = Input(shape=(seq_len,))
    embededding_layer = Embedding(vocab_size, emb_dim)
    lstm_layer = LSTM(hid_dim, return_sequences=True, return_state=True)

    # connect layer
    # TODO: LSTMレイヤの多層化、双方向化
    outputs, _, _ = lstm_layer(embededding_layer(input_layer), initial_state=encoder_states)

    return input_layer, outputs

# attention TODO

# generator
#   model : (en_seq_len, ja_seq_len) -> (ja_seq_len, ja_vocab_size)
def generator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size, emb_dim=256, hid_dim=256):

    encoder_inputs, _, encoder_states = encoder(en_seq_len, en_vocab_size, emb_dim, hid_dim)
    decoder_inputs, decoder_outputs = decoder(encoder_states, ja_seq_len, ja_vocab_size, emb_dim=256, hid_dim=256)

    dense_layer = Dense(ja_vocab_size, activation='softmax')

    model = Model([encoder_inputs, decoder_inputs], dense_layer(decoder_outputs))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model, encoder_inputs, decoder_inputs
