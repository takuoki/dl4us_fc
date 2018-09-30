# 05_main

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

discriminator_model, in3 = discriminator(en_seq_len, ja_seq_len, en_vocab_size, ja_vocab_size)
plot_model(discriminator_model, to_file=final_dir+'discriminator_model.png')

gan_model = combined_models(generator_model, discriminator_model, in1, in2, in3, en_seq_len, ja_seq_len)
plot_model(gan_model, to_file=final_dir+'gan_model.png')

# train
train(generator_model, discriminator_model, gan_model, ja_seq_len, ja_vocab_size, x_train, y_train, step=10, batch_size=32)

# TODO: 学習結果(history)をplot

# validデータの翻訳 & 保存
test_count = 5

en_seqs = detoken(detokenizer_en, x_valid[:test_count])
ja_pred_seqs = detoken(detokenizer_ja, predict_all(generator_model, x_valid[:test_count], ja_seq_len))
ja_seqs = detoken(detokenizer_ja, y_valid[:test_count])

for i in range(test_count):
    print('EN(', i, '): ', en_seqs[i])
    print('JA-predict(', i, '): ', ja_pred_seqs[i])
    print('JA-answer (', i, '): ', ja_seqs[i])

# TODO: 精度検証