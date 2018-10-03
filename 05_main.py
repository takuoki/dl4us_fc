# 05_main

%matplotlib inline
import matplotlib.pyplot as plt
from keras.utils import plot_model

final_dir = '/root/userspace/final/'

# data load
x_train, y_train, x_valid, y_valid, tokenizer_en, tokenizer_ja, detokenizer_en, detokenizer_ja = load_data(final_dir)

en_seq_len = 0
for i in range(x_train.shape[0]):
    if x_train[i].shape[0] > en_seq_len:
        en_seq_len = x_train[i].shape[0]
for i in range(x_valid.shape[0]):
    if x_valid[i].shape[0] > en_seq_len:
        en_seq_len = x_valid[i].shape[0]

ja_seq_len = 0
for i in range(y_train.shape[0]):
    if y_train[i].shape[0] > ja_seq_len:
        ja_seq_len = y_train[i].shape[0]
for i in range(y_valid.shape[0]):
    if y_valid[i].shape[0] > ja_seq_len:
        ja_seq_len = y_valid[i].shape[0]

en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1

# 各Modelを取得
generator_model, in1, in2 = generator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
plot_model(generator_model, to_file=final_dir+'generator_model.png')

generator_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

discriminator_model, in3 = discriminator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
plot_model(discriminator_model, to_file=final_dir+'discriminator_model.png')

# train
history = train(
    ja_seq_len, ja_vocab_size, x_train, y_train,
    x_valid, y_valid, detokenizer_en, detokenizer_ja,
    epochs=20, batch_size=128)

# historyをplot
plt.title('acc/loss')
plt.plot(history['gen_history'].history['acc'])
plt.plot(history['gen_history'].history['loss'])
plt.plot(history['disc_history'].history['acc'])
plt.plot(history['disc_history'].history['loss'])
plt.legend(['gen_acc', 'gen_loss', 'disc_acc', 'disc_loss'])
plt.show

# validデータの翻訳 & 保存
translate(detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len, ja_vocab_size)

# TODO: 精度検証
