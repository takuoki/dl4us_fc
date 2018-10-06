# step3

%matplotlib inline
import matplotlib.pyplot as plt
from datetime import datetime as dt
from keras.utils import plot_model
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, concatenate
from keras.layers.core import Dense, Lambda
from keras import backend as K

# bidirectional_decoder
#   input_layer : input layer for ja sentense (size=(seq_len,))
#   outputs     : decoded sentense (size=(seq_len, hid_dim))
def bidirectional_decoder(encoder_states, seq_len, vocab_size, emb_dim=256, hid_dim=256):

    # layer
    input_layer = Input(shape=(seq_len,))
    embededding_layer = Embedding(vocab_size, emb_dim)
    lstm_layer = LSTM(hid_dim, return_sequences=True)
    rev_lstm_layer = LSTM(hid_dim, return_sequences=True, go_backwards=True)

    # connect layer
    x = embededding_layer(input_layer)
    x1 = lstm_layer(x, initial_state=encoder_states)
    x2 = rev_lstm_layer(x, initial_state=encoder_states)
    x2 = Lambda(lambda x: K.reverse(x, axes=1))(x2)
    outputs = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([x1, x2])

    return input_layer, outputs

# bidirectional_generator
#   model : (en_seq_len, ja_seq_len) -> (ja_seq_len, ja_vocab_size)
def bidirectional_generator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size, emb_dim=256, hid_dim=256):

    encoder_inputs, _, encoder_states = encoder(en_seq_len, en_vocab_size, emb_dim, hid_dim)
    decoder_inputs, decoder_outputs = bidirectional_decoder(encoder_states, ja_seq_len, ja_vocab_size, emb_dim=256, hid_dim=256)

    dense_layer = Dense(ja_vocab_size, activation='softmax')

    model = Model([encoder_inputs, decoder_inputs], dense_layer(decoder_outputs))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

# *** main logic ***

final_dir = '/root/userspace/final/'

# create output dir
outdir = final_dir+'out/'+dt.now().strftime('%Y%m%d%H%M%S')
os.makedirs(outdir)

# data load
x_train, y_train, x_valid, y_valid, tokenizer_en, tokenizer_ja, detokenizer_en, detokenizer_ja = load_data(final_dir)

en_seq_len = x_train.shape[1]
ja_seq_len = y_train.shape[1]

en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1

# 各Modelを取得
bd_generator_model = bidirectional_generator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
plot_model(bd_generator_model, to_file=outdir+'/bd_generator_model.png')

# train generator
epochs = 30
batch_size = 128
gen_history = train_generator(bd_generator_model, x_train, y_train, x_valid, y_valid, epochs=epochs, batch_size=batch_size)
print('done train_generator')

save_model(outdir, bd_generator_model, 'bd_generator_model')
save_pickle(outdir, gen_history.history, 'gen_history')

print('sample translate')
translate_sample(bd_generator_model, detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len)

print('sample translate with each steps')
test_count = 3
pred_seqs = predict_all(bd_generator_model, x_valid[:test_count], ja_seq_len, return_each=True)
for test_case in range(pred_seqs.shape[1]):
    for i in range(ja_seq_len):
        print('case', test_case, '(', i, '):', detoken(detokenizer_ja, pred_seqs[i][test_case:test_case+1])[0])

# historyをplot
plt.title('acc/loss')
plt.plot(bd_generator_model.history['acc'])
plt.plot(bd_generator_model.history['val_acc'])
plt.plot(bd_generator_model.history['loss'])
plt.plot(bd_generator_model.history['val_loss'])
plt.legend([
    'g_acc', 'g_v_acc',
    'g_loss', 'g_v_loss',
])
plt.show

# validデータの予測 & 保存
y_valid_wp, y_pred_wp = save_all_prediction(bd_generator_model, outdir, 'valid', x_valid, y_valid, ja_seq_len)

# 精度検証
score = scoreBLEU(detokenizer_ja, y_pred_wp, y_valid_wp)

# 結果の保存
save_result(outdir, [['BLEU score', score]])
print('done')
