# 04_train

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# def train_only_generator(ja_seq_len, x_train, y_train, epochs=100, batch_size=128):

#     en_train = x_train
#     ja_train = y_train

#     # 1文字前にスライドしたデータがターゲット
#     ja_target = np.hstack((ja_train[:, 1:], np.zeros((len(ja_train), 1), dtype=np.int32)))

#     hist = generator_model.fit(
#         [en_train, ja_train], 
#         np.expand_dims(ja_target, -1), 
#         batch_size=batch_size, 
#         epochs=epochs)

#     return hist

# train_discriminator
def train_discriminator(ja_seq_len, x_train, y_train, batch_size=128):

    switch_trainable(discriminator_model, True)

    # batch_size数だけ、本物の英語文/日本語文をランダムに選択
    test_size = x_train.shape[0] - batch_size
    en_train, _, ja_train, _ = train_test_split(x_train, y_train, test_size=test_size)

    # 英語文を元に日本語文を生成
    ja_generated = predict_all(en_train, ja_seq_len)

    # 本物/偽物データを結合
    en_input = np.concatenate((en_train, en_train))
    ja_input = np.array(np.concatenate((ja_train, ja_generated)), dtype=np.int32)
    y = np.zeros([2*batch_size, 2], dtype=np.int32)
    y[:batch_size, 1] = 1
    y[batch_size:, 0] = 1

    # discriminatorの学習
    return discriminator_model.train_on_batch([en_input, ja_input], y)

# train generator
def train_generator(ja_seq_len, x_train, batch_size=128):

    switch_trainable(discriminator_model, False)

    # batch_size数だけ、学習用の英語文をランダムに選択
    test_size = x_train.shape[0] - batch_size
    en_train, _, _, _ = train_test_split(x_train, x_train, test_size=test_size)

    # batch_size数だけ、日本語文初期値データを生成
    ja_initial_seqs = initialize_seq(np.zeros((batch_size, ja_seq_len), dtype=np.int32))

    # 正解ラベルを用意
    y = np.zeros([batch_size, 2], dtype=np.int32)
    y[:, 1] = 1

    # ganを学習
    return gan_model.train_on_batch([en_train, ja_initial_seqs, en_train], y)

# train
def train(
    ja_seq_len,
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

    # loop (=epoch)
    for i in tqdm(range(step)):
        disc_history.append(train_discriminator(ja_seq_len, x_train, y_train, batch_size))
        gan_history.append(train_generator(ja_seq_len, x_train, batch_size))

        # 途中結果の確認
        if i % 10 == 9:
            translate(detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len, test_count=2)
            # TODO: 途中結果の保存（generator/discriminatorの重み、学習結果(history)）

    return {'disc_history': disc_history, 'gan_history': gan_history}
