# 05_test

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

# for through pass test
x_train = x_train[:500]
y_train = y_train[:500]
x_valid = x_valid[:250]
y_valid = y_valid[:250]

en_seq_len = x_train.shape[1]
ja_seq_len = y_train.shape[1]

en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1

# 各Modelを取得
generator_model, ec_model, dc_model = generator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
plot_model(generator_model, to_file=outdir+'/generator_model.png')
plot_model(ec_model, to_file=outdir+'/ec_model.png')
plot_model(dc_model, to_file=outdir+'/dc_model.png')

discriminator_model = discriminator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
plot_model(discriminator_model, to_file=outdir+'/discriminator_model.png')

epochs = 3
batch_size = 128

# train generator
gen_history = train_generator(generator_model, x_train, y_train, x_valid, y_valid, epochs=epochs, batch_size=batch_size)
print('done train_generator')

save_model(outdir, generator_model, 'generator_model')
save_model(outdir, ec_model, 'ec_model')
save_model(outdir, dc_model, 'dc_model')
save_pickle(outdir, gen_history.history, 'gen_history')

print('sample translate')
translate_sample(generator_model, detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len)

print('sample translate with each steps')
test_count = 3
pred_seqs = predict_all(generator_model, x_valid[:test_count], ja_seq_len, return_each=True)
for test_case in range(pred_seqs.shape[1]):
    for i in range(ja_seq_len):
        print('case', test_case, '(', i, '):', detoken(detokenizer_ja, pred_seqs[i][test_case:test_case+1])[0])

print('sample translate for 1 word generator')
translate_sample_for1(ec_model, dc_model, detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len)

# train discriminator
print('start train_discriminator')
disc_history = train_discriminator(discriminator_model, generator_model, ja_seq_len, x_train, y_train, x_valid, y_valid, epochs=epochs, batch_size=batch_size)
print('done train_discriminator')

save_pickle(outdir, disc_history.history, 'disc_history')
save_model(outdir, discriminator_model, 'discriminator_model')

# historyをplot
plt.title('acc/loss')
plt.plot(gen_history.history['acc'])
plt.plot(gen_history.history['val_acc'])
plt.plot(gen_history.history['loss'])
plt.plot(gen_history.history['val_loss'])
plt.plot(disc_history.history['acc'])
plt.plot(disc_history.history['val_acc'])
plt.plot(disc_history.history['loss'])
plt.plot(disc_history.history['val_loss'])
plt.legend([
    'g_acc', 'g_v_acc',
    'g_loss', 'g_v_loss',
    'd_acc', 'd_v_acc',
    'd_loss', 'd_v_loss',
])
plt.show

# validデータの予測 & 保存
y_valid_wp, y_pred_wp = save_all_prediction(generator_model, outdir, 'valid', x_valid, y_valid, ja_seq_len)
y_valid_wp_for1, y_pred_wp_for1 = save_all_prediction_for1(ec_model, dc_model, outdir, 'valid_for1', x_valid, y_valid, ja_seq_len)

# 精度検証
score = scoreBLEU(detokenizer_ja, y_pred_wp, y_valid_wp)
score_for1 = scoreBLEU(detokenizer_ja, y_pred_wp_for1, y_valid_wp_for1)

# 結果の保存
save_result(outdir, [['BLEU score', score], ['BLEU score for1', score_for1]])
print('done')
