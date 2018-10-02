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

# generator_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

discriminator_model, in3 = discriminator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
plot_model(discriminator_model, to_file=final_dir+'discriminator_model.png')

gan_model = combined_models(in1, in2, in3, en_seq_len, ja_seq_len)
plot_model(gan_model, to_file=final_dir+'gan_model.png')

# train
# history = train_only_generator(ja_seq_len, x_train, y_train, epochs=50, batch_size=32)
history = train(
    ja_seq_len, ja_vocab_size, x_train, y_train,
    x_valid, y_valid, detokenizer_en, detokenizer_ja,
    step=50, batch_size=128)

# historyをplot
plt.title('loss')
plt.plot(history.history['loss'])
plt.legend(['generator loss'])
plt.show

# validデータの翻訳 & 保存
translate(detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len, ja_vocab_size)

# TODO: 精度検証