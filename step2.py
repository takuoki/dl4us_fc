# step2

%matplotlib inline
import matplotlib.pyplot as plt
from datetime import datetime as dt
from keras.utils import plot_model
import os

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
generator_model_10 = generator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
generator_model_20 = generator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
generator_model_50 = generator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)

discriminator_model_10 = discriminator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
discriminator_model_20 = discriminator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
discriminator_model_50 = discriminator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)

# train
epochs = 30 # for discriminator
batch_size = 128

train_generator(generator_model_10, x_train, y_train, x_valid, y_valid, epochs=10, batch_size=batch_size)
save_model(outdir, generator_model_10, 'generator_model_10')
print('done train generator 10 (↓sample generated sentence)')
translate_sample(generator_model_10, detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len)

train_generator(generator_model_20, x_train, y_train, x_valid, y_valid, epochs=20, batch_size=batch_size)
save_model(outdir, generator_model_20, 'generator_model_20')
print('done train generator 20 (↓sample generated sentence)')
translate_sample(generator_model_20, detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len)

train_generator(generator_model_50, x_train, y_train, x_valid, y_valid, epochs=50, batch_size=batch_size)
save_model(outdir, generator_model_50, 'generator_model_50')
print('done train generator 50 (↓sample generated sentence)')
translate_sample(generator_model_50, detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len)

print('start train discriminator')

disc_history_10 = train_discriminator(discriminator_model_10, generator_model_10, ja_seq_len, x_train, y_train, x_valid, y_valid, epochs=epochs, batch_size=batch_size)
save_model(outdir, discriminator_model_10, 'discriminator_model_10')
save_pickle(outdir, disc_history_10.history, 'disc_history_10')

print('done train discriminator 10')

disc_history_20 = train_discriminator(discriminator_model_20, generator_model_20, ja_seq_len, x_train, y_train, x_valid, y_valid, epochs=epochs, batch_size=batch_size)
save_model(outdir, discriminator_model_20, 'discriminator_model_20')
save_pickle(outdir, disc_history_20.history, 'disc_history_20')

print('done train discriminator 20')

disc_history_50 = train_discriminator(discriminator_model_50, generator_model_50, ja_seq_len, x_train, y_train, x_valid, y_valid, epochs=epochs, batch_size=batch_size)
save_model(outdir, discriminator_model_50, 'discriminator_model_50')
save_pickle(outdir, disc_history_50.history, 'disc_history_50')

print('done train discriminator 50')

# historyをplot
plt.title('acc/loss')
plt.plot(disc_history_10.history['acc'])
plt.plot(disc_history_10.history['loss'])
plt.plot(disc_history_20.history['acc'])
plt.plot(disc_history_20.history['loss'])
plt.plot(disc_history_50.history['acc'])
plt.plot(disc_history_50.history['loss'])
plt.legend(['acc_10', 'loss_10', 'acc_20', 'loss_20', 'acc_50', 'loss_50'])
plt.show

# 結果の保存
save_result(outdir, 0)
print('done')
