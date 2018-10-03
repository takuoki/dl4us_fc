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
#   input_layer : input layer for one ja character (size=(seq_len, vocab_size))
#   outputs     : encoded sentense (size=(seq_len, hid_dim))
def decoder(encoder_states, seq_len, vocab_size, emb_dim=256, hid_dim=256):

    # layer
    input_layer = Input(shape=(seq_len, vocab_size))
    embededding_layer = MyEmbedding(vocab_size, emb_dim)
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

    return model, encoder_inputs, decoder_inputs

# initialize_seq
def initialize_seq(ja_seqs):

    results = np.array([np.zeros(ja_seqs.shape[1], dtype=np.int32)])
    for i in range(ja_seqs.shape[0]):
        # 1文字目bos、以降はひとつスライド
        results = np.concatenate((results, np.array([np.append(np.array([1]), ja_seqs[i][:-1])])))

    return np.array(results[1:], dtype=np.int32)

# predict
def predict(en_input_seqs, ja_input_seqs, ja_vocab_size):

    ja_input_onehot_seqs = onehot(ja_input_seqs, ja_vocab_size)
    outputs = generator_model.predict([en_input_seqs[0:2], ja_input_onehot_seqs[0:2]])
    for i in range(int(en_input_seqs.shape[0]/2-1)):
        outputs = np.concatenate((outputs, generator_model.predict([en_input_seqs[i+2:i+4], ja_input_onehot_seqs[i+2:i+4]])))

    return np.argmax(outputs, axis=-1)

# predict_all
def predict_all(en_input_seqs, ja_seq_len, ja_vocab_size):

    ja_output_seqs = np.zeros((en_input_seqs.shape[0], ja_seq_len), dtype=np.int32)
    for _ in range(ja_seq_len):
        ja_input_seqs = initialize_seq(ja_output_seqs)
        ja_output_seqs = predict(en_input_seqs, ja_input_seqs, ja_vocab_size)

    return ja_output_seqs
