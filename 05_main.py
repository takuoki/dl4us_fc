# 05_main

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
generator_model, in1, in2 = generator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
plot_model(generator_model, to_file=outdir+'/generator_model.png')

discriminator_model, in3 = discriminator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
plot_model(discriminator_model, to_file=outdir+'/discriminator_model.png')

save_model(outdir, 'start')

# train
epochs = 20
batch_size = 128
gen_history = train_only_generator(ja_seq_len, ja_vocab_size, x_train, y_train, epochs=epochs, batch_size=batch_size)
print('done train_only_generator')

save_model(outdir, 'end', gen=True, disc=False)
save_pickle(outdir, gen_history, 'gen_history')
translate_sample(detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len)

print('start train_discriminator')
disc_history = train_discriminator(ja_seq_len, x_train, y_train, epochs=epochs, batch_size=batch_size)
print('done train_discriminator')

save_pickle(outdir, disc_history, 'disc_history')
save_model(outdir, 'end', gen=False, disc=True)

# historyをplot
plt.title('acc/loss')
plt.plot(gen_history.history['acc'])
plt.plot(gen_history.history['loss'])
plt.plot(disc_history.history['acc'])
plt.plot(disc_history.history['loss'])
plt.legend(['gen_acc', 'gen_loss', 'disc_acc', 'disc_loss'])
plt.show

# validデータの予測 & 保存
y_valid_wp, y_pred_wp = save_all_prediction(outdir, x_valid, y_valid, ja_seq_len)

# 精度検証
score = scoreBLEU(detokenizer_ja, y_pred_wp, y_valid_wp)

# 結果の保存
save_result(outdir, score)
