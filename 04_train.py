# 04_train

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def train_only_generator(ja_seq_len, ja_vocab_size, x_train, y_train, epochs=20, batch_size=2):

    en_train = x_train
    ja_train = y_train

    # 1文字前にスライドしたデータがターゲット
    ja_target = np.hstack((ja_train[:, 1:], np.zeros((len(ja_train), 1), dtype=np.int32)))

    history = []
    for _ in tqdm(range(epochs)):
        for i in range(int(en_train.shape[0]/batch_size)):
            print(i)
            # onehot表現のデータを取得
            from_i = i*batch_size
            to_i = (i+1)*batch_size
            ja_train_onehot = onehot(ja_train[from_i:to_i], ja_vocab_size)
            ja_target_onehot = onehot(ja_target[from_i:to_i], ja_vocab_size)

            history.append(generator_model.train_on_batch([en_train[from_i:to_i], ja_train_onehot], ja_target_onehot))

    print('done train_only_generator')

    return history

# train_discriminator
def train_discriminator(ja_seq_len, ja_vocab_size, x_train, y_train, detokenizer_ja, batch_size=128):

    print('start train_discriminator')
    switch_trainable(discriminator_model, True)

    # batch_size数だけ、本物の英語文/日本語文をランダムに選択
    test_size = x_train.shape[0] - batch_size
    en_train, _, ja_train, _ = train_test_split(x_train, y_train, test_size=test_size)

    # 英語文を元に日本語文を生成
    ja_generated = predict_all(en_train, ja_seq_len, ja_vocab_size)
    print('ja_generated[0]:', detoken(detokenizer_ja, ja_generated[:1]))

    # 本物/偽物データを結合
    en_input = np.concatenate((en_train, en_train))
    ja_input = initialize_seq(np.array(np.concatenate((ja_train, ja_generated)), dtype=np.int32))
    ja_input_onehot = onehot(ja_input, ja_vocab_size)
    y = np.zeros([2*batch_size, 2], dtype=np.int32)
    y[:batch_size, 1] = 1
    y[batch_size:, 0] = 1

    # discriminatorの学習
    history = []
    for i in range(batch_size):
        history.append(discriminator_model.train_on_batch([en_input[i:i+2], ja_input_onehot[i:i+2]], y[i:i+2]))

    print('done train_discriminator')

    return history

# train generator
def train_generator(ja_seq_len, ja_vocab_size, x_train, batch_size=128):

    print('start train_generator')
    switch_trainable(discriminator_model, False)

    # batch_size数だけ、学習用の英語文をランダムに選択
    test_size = x_train.shape[0] - batch_size
    en_train, _, _, _ = train_test_split(x_train, x_train, test_size=test_size)

    # batch_size数だけ、日本語文初期値データを生成
    ja_initial_seqs = np.zeros((batch_size, ja_seq_len), dtype=np.int32)
    ja_initial_seqs[:, 0] = 1
    ja_initial_seqs_onehot = onehot(ja_initial_seqs, ja_vocab_size)

    # 正解ラベルを用意
    y = np.zeros([batch_size, 2], dtype=np.int32)
    y[:, 1] = 1

    # ganを学習
    history = []
    for i in range(int(batch_size/2)):
        history.append(gan_model.train_on_batch([en_train[i:i+2], ja_initial_seqs_onehot[i:i+2], en_train[i:i+2]], y[i:i+2]))

    print('done train_generator')

    return history

# train
def train(
    ja_seq_len,
    ja_vocab_size,
    x_train,
    y_train,
    x_valid,
    y_valid,
    detokenizer_en,
    detokenizer_ja,
    step=1000,
    batch_size=128):

    disc_history = []
    gan_history = []

    # 最初にSeq2Seq（generator）だけである程度学習しておく
    gen_history = train_only_generator(ja_seq_len, ja_vocab_size, x_train, y_train)

    translate(detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len, ja_vocab_size, test_count=2)

    # loop (=epoch)
    for i in tqdm(range(step)):
        disc_history.append(train_discriminator(ja_seq_len, ja_vocab_size, x_train, y_train, detokenizer_ja, batch_size))
        gan_history.append(train_generator(ja_seq_len, ja_vocab_size, x_train, batch_size))

        # 途中結果の確認
        if i % 10 == 9:
            translate(detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len, ja_vocab_size, test_count=2)
            # TODO: 途中結果の保存（generator/discriminatorの重み、学習結果(history)）

    return {'gen_history': gen_history,'disc_history': disc_history, 'gan_history': gan_history}
