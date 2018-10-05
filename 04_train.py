# 04_train

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def train_generator(generator_model, x_train, y_train, x_valid, y_valid, epochs=20, batch_size=128):

    en_train = x_train
    ja_train = y_train
    en_valid = x_valid
    ja_valid = y_valid

    # 1文字前にスライドしたデータがターゲット
    ja_target = np.hstack((ja_train[:, 1:], np.zeros((len(ja_train), 1), dtype=np.int32)))
    ja_target = np.expand_dims(ja_target, -1)

    ja_val_target = np.hstack((ja_valid[:, 1:], np.zeros((len(ja_valid), 1), dtype=np.int32)))
    ja_val_target = np.expand_dims(ja_val_target, -1)

    return generator_model.fit([en_train, ja_train], ja_target, validation_data=([en_valid, ja_valid], ja_val_target), epochs=epochs, batch_size=batch_size)

# train_discriminator
def train_discriminator(discriminator_model, generator_model, ja_seq_len, x_train, y_train, x_valid, y_valid, epochs=20, batch_size=128):

    # データの半分だけ、本物の英語文/日本語文をランダムに選択（validデータはそのまま）
    test_size = int(x_train.shape[0] / 2)
    en_train, _, ja_train, _ = train_test_split(x_train, y_train, test_size=test_size)
    en_valid = x_valid
    ja_valid = y_valid

    # 英語文を元に日本語文を生成
    ja_generated = predict_all(generator_model, en_train, ja_seq_len)
    ja_generated = initialize_seq(ja_generated)
    ja_val_generated = predict_all(generator_model, en_valid, ja_seq_len)
    ja_val_generated = initialize_seq(ja_val_generated)

    # 本物/偽物データを結合
    en_input = np.concatenate((en_train, en_train))
    ja_input = np.concatenate((ja_train, ja_generated))
    y = np.zeros([2*test_size, 2], dtype=np.int32)
    y[:test_size, 1] = 1
    y[test_size:, 0] = 1

    en_val_input = np.concatenate((en_valid, en_valid))
    ja_val_input = np.concatenate((ja_valid, ja_val_generated))
    y_val = np.zeros([2*en_valid.shape[0], 2], dtype=np.int32)
    y_val[:en_valid.shape[0], 1] = 1
    y_val[en_valid.shape[0]:, 0] = 1

    # discriminatorの学習
    return discriminator_model.fit([en_input, ja_input], y, validation_data=([en_val_input, ja_val_input], y_val), epochs=epochs, batch_size=batch_size)
