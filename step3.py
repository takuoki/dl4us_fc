# step3

%matplotlib inline
import matplotlib.pyplot as plt
from datetime import datetime as dt
from keras.utils import plot_model
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Average
from keras.layers.core import Dense, Lambda
from keras import backend as K

# bidirectional_decoder
#   input_layer : input layer for ja sentense (size=(seq_len,))
#   outputs     : decoded sentense (size=(seq_len, hid_dim))
def bidirectional_decoder(encoder_states, seq_len, vocab_size, emb_dim=256, hid_dim=256):

    # layer
    input_layer = Input(shape=(seq_len,))
    embededding_layer = Embedding(vocab_size, emb_dim, mask_zero=True)
    lstm_layer = LSTM(hid_dim, return_sequences=True)
    rev_lstm_layer = LSTM(hid_dim, return_sequences=True, go_backwards=True)

    # connect layer
    x = embededding_layer(input_layer)
    x1 = lstm_layer(x, initial_state=encoder_states)
    x2 = rev_lstm_layer(x, initial_state=encoder_states)
    x2 = Lambda(lambda x: K.reverse(x, axes=1))(x2)
    # outputs = Average()([x1, x2])
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
# epochs = 30
# batch_size = 128
# gen_history = train_generator(bd_generator_model, x_train, y_train, x_valid, y_valid, epochs=epochs, batch_size=batch_size)
# gen_history = gen_history.history
# print('done train_generator')

# save_model(outdir, bd_generator_model, 'bd_generator_model')
# save_pickle(outdir, gen_history, 'gen_history')
load_model(final_dir+'out/20181008015838', bd_generator_model, 'bd_generator_model')
gen_history = load_pickle(final_dir+'out/20181008015838', 'gen_history')

# historyをplot
plt.title('acc/loss')
plt.plot(gen_history['acc'])
plt.plot(gen_history['val_acc'])
plt.plot(gen_history['loss'])
plt.plot(gen_history['val_loss'])
plt.legend([
    'g_acc', 'g_v_acc',
    'g_loss', 'g_v_loss',
])
plt.show

# 通常のgeneratorモデルで予測したvalidデータを読み込む
preded_outdir = '20181006031811'
pred_seq = load_csv(final_dir+'out/'+preded_outdir, 'valid', ja_seq_len)

# 今回学習したモデルで予測結果を出力
# y_valid_wp, y_pred_wp = save_all_prediction(bd_generator_model, outdir, 'valid', x_valid, y_valid, ja_seq_len)

# 今回学習したモデルで予測結果を変換（改良のつもり）
y_valid_wp, y_pred_wp = save_prediction(bd_generator_model, outdir, 'valid', x_valid, pred_seq, y_valid, ja_seq_len)

# 精度検証
score = scoreBLEU(detokenizer_ja, y_pred_wp, y_valid_wp)

# 結果の保存
save_result(outdir, [['BLEU score', score], ['detail', 'add mask_zero=True to decoder']])

# 結果の表示
cnt = 10
pred1 = detoken(detokenizer_ja, pred_seq[:cnt, 1:], join_char=True)
pred2 = detoken(detokenizer_ja, y_pred_wp[:cnt], join_char=True)
valid = detoken(detokenizer_ja, y_valid_wp[:cnt], join_char=True)

for i in range(cnt):
    print('pred1(', i, '):', pred1[i])
    print('pred2(', i, '):', pred2[i])
    print('valid(', i, '):', valid[i])

print('done')
