# 04_train

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# train_discriminator
def train_discriminator(generator_model, discriminator_model, ja_seq_len, ja_vocab_size, x_train, y_train, batch_size=128):

    switch_trainable(discriminator_model, True)

    # batch_size数だけ、本物の英語文/日本語文をランダムに選択
    test_size = x_train.shape[0] - batch_size
    en_train, _, ja_train, _ = train_test_split(x_train, y_train, test_size=test_size)

    en_train_xn = en_train
    for _ in range(ja_seq_len-1):
        en_train_xn = np.append(en_train_xn, en_train, axis=0)
    ja_train_xn = ja_train
    for i in range(ja_seq_len-1):
        # TODO: i+1以降をpaddingに変換する
        ja_train_xn = np.append(ja_train_xn, ja_train, axis=0)

    # batch_size x ja_seq_len数だけ、偽物の日本語文を生成
    ja_generated = np.array([np.zeros(ja_seq_len)])
    for i in range(batch_size):
        ja_output_seq = np.array([np.zeros(ja_seq_len)])
        for _ in range(ja_seq_len):
            ja_input_seq = initialize_seq(ja_output_seq)
            ja_output_seq = predict(generator_model, en_train[i:i+1], ja_input_seq)
            ja_generated = np.append(ja_generated, ja_output_seq, axis=0)
    ja_generated = np.array(ja_generated[1:], dtype='int32')

    # 本物/偽物データを結合
    en_input = np.concatenate((en_train_xn, en_train_xn))
    ja_input = np.concatenate((ja_train_xn, ja_generated))
    ja_input_onehot = onehot(ja_input, ja_vocab_size)
    y = np.zeros([2*batch_size*ja_seq_len, 2])
    y[:batch_size*ja_seq_len, 1] = 1
    y[batch_size*ja_seq_len:, 0] = 1

    # discriminatorの学習
    print('en_input:', en_input.shape)
    print('ja_input_onehot:', ja_input_onehot.shape)
    print('y:', y.shape)
    discriminator_model.train_on_batch([en_input, ja_input_onehot], y)

    # TODO: 学習後の損失関数の値を返す

# train generator
def train_generator(discriminator_model, gan_model, ja_seq_len, x_train, batch_size=128):

    switch_trainable(discriminator_model, False)

    # batch_size数だけ、学習用の英語文をランダムに選択
    test_size = x_train.shape[0] - batch_size
    en_train, _, _, _ = train_test_split(x_train, x_train, test_size=test_size)

    # TODO: 学習用の英語文データをja_seq_len数に増幅
    en_train_xn = en_train

    # batch_size x ja_seq_len数だけ、日本語文初期値データを生成
    ja_generated = []
    for i in range(batch_size):
        ja_input_seq = initial_seq(ja_seq_len)
        for _ in range(ja_seq_len):
            ja_input_seq = predict(generator_model, en_train[i], ja_input_seq)
            ja_generated = append(ja_generated, ja_input_seq)

    # 正解ラベルを用意
    y = np.zeros([batch_size*ja_seq_len, 2])
    y[:, 1] = 1

    # ganを学習
    gan_model.train_on_batch([en_train_xn, ja_generated], y)

    # TODO: 学習後の損失関数の値を返す

# train
def train(generator_model, discriminator_model, gan_model, ja_seq_len, ja_vocab_size, x_train, y_train, step=1000, batch_size=128):

    # TODO: 学習時の損失関数の値を格納する箱を用意

    # loop (=epoch)
    for _ in tqdm(range(step)):
        train_discriminator(generator_model, discriminator_model, ja_seq_len, ja_vocab_size, x_train, y_train, batch_size)
        train_generator(discriminator_model, gan_model, ja_seq_len, x_train, batch_size)

        # TODO: 途中結果の保存（generator/discriminatorの重み、学習結果(history)）と、サンプル出力(500回ごととか)
