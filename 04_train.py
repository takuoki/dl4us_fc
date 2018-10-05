# 04_train

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def train_only_generator(ja_seq_len, ja_vocab_size, x_train, y_train, epochs=20, batch_size=128):

    en_train = x_train
    ja_train = y_train

    # 1文字前にスライドしたデータがターゲット
    ja_target = np.hstack((ja_train[:, 1:], np.zeros((len(ja_train), 1), dtype=np.int32)))
    ja_target = np.expand_dims(ja_target, -1)

    return generator_model.fit([en_train, ja_train], ja_target, epochs=epochs, batch_size=batch_size)

# train_discriminator
def train_discriminator(ja_seq_len, x_train, y_train, epochs=20, batch_size=128):

    # データの半分だけ、本物の英語文/日本語文をランダムに選択
    test_size = int(x_train.shape[0] / 2)
    en_train, _, ja_train, _ = train_test_split(x_train, y_train, test_size=test_size)

    # 英語文を元に日本語文を生成
    ja_generated = predict_all(en_train, ja_seq_len)
    ja_generated = initialize_seq(ja_generated)

    # 本物/偽物データを結合
    en_input = np.concatenate((en_train, en_train))
    ja_input = np.concatenate((ja_train, ja_generated))
    y = np.zeros([2*test_size, 2], dtype=np.int32)
    y[:test_size, 1] = 1
    y[test_size:, 0] = 1

    # discriminatorの学習
    return discriminator_model.fit([en_input, ja_input], y, epochs=epochs, batch_size=batch_size)
