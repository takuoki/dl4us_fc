# 04_train

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def train_only_generator(ja_seq_len, x_train, y_train, epochs=100, batch_size=128):

    # 英語文をja_seq_len数に増幅（en_train[k] == en_train[k+ja_seq_len*i]）
    en_train = x_train
    # for _ in range(ja_seq_len-1):
    #     en_train = np.append(en_train, x_train, axis=0)

    # 日本語文をja_seq_len数に増幅（ja_train[k] == ja_train[k+ja_seq_len*i]）
    # ※オリジナルデータは同じだが、(i+1)文字目以降はpaddingに変換されている
    # ja_train = np.array([np.zeros(ja_seq_len)])
    # for i in range(ja_seq_len):
    #     for j in range(y_train.shape[0]): # batch_size loop
    #         # i+1以降をpaddingに変換
    #         ja_seq_pad = np.append(y_train[j][0:i+1], np.zeros(ja_seq_len-i-1))
    #         ja_train = np.append(ja_train, np.array([ja_seq_pad]), axis=0)
    # ja_train = np.array(ja_train[1:], dtype='int32')
    ja_train = y_train

    # 1文字前にスライドしたデータがターゲット
    ja_target = np.hstack((ja_train[:, 1:], np.zeros((len(ja_train), 1), dtype=np.int32)))

    hist = generator_model.fit(
        [en_train, ja_train], 
        np.expand_dims(ja_target, -1), 
        batch_size=batch_size, 
        epochs=epochs)

    return hist

# train_discriminator
def train_discriminator(ja_seq_len, ja_vocab_size, x_train, y_train, batch_size=128):

    switch_trainable(discriminator_model, True)

    # batch_size数だけ、本物の英語文/日本語文をランダムに選択
    test_size = x_train.shape[0] - batch_size
    en_train, _, ja_train, _ = train_test_split(x_train, y_train, test_size=test_size)

    # ja_seq_len数に増幅
    # en_train_xn = en_train
    # for _ in range(ja_seq_len-1):
    #     en_train_xn = np.append(en_train_xn, en_train, axis=0)
    # ja_train_xn = np.array([np.zeros(ja_seq_len)])
    # for i in range(ja_seq_len):
    #     for j in range(ja_train.shape[0]): # batch_size loop
    #         # BOSを削除
    #         ja_seq_rm_bos = np.append(ja_train[j][1:], np.zeros(1))
    #         # i+1以降をpaddingに変換
    #         ja_seq_pad = np.append(ja_seq_rm_bos[0:i+1], np.zeros(ja_seq_len-i-1))
    #         ja_train_xn = np.append(ja_train_xn, np.array([ja_seq_pad]), axis=0)
    # ja_train_xn = np.array(ja_train_xn[1:], dtype='int32')

    # batch_size x ja_seq_len数だけ、偽物の日本語文を生成
    # ja_generated = np.array([np.zeros(ja_seq_len)])
    # for i in range(batch_size):
    #     ja_output_seq = np.array([np.zeros(ja_seq_len)])
    #     for _ in range(ja_seq_len):
    #         ja_input_seq = initialize_seq(ja_output_seq)
    #         ja_output_seq = predict(en_train[i:i+1], ja_input_seq)
    #         ja_generated = np.append(ja_generated, ja_output_seq, axis=0)
    # ja_generated = np.array(ja_generated[1:], dtype='int32')
    ja_generated = predict_all(en_train, ja_seq_len, ja_vocab_size)

    # 本物/偽物データを結合
    en_input = np.concatenate((en_train, en_train))
    ja_input = np.array(np.concatenate((ja_train, ja_generated)), dtype=np.int32)
    ja_input_onehot = onehot(ja_input, ja_vocab_size)
    y = np.zeros([2*batch_size, 2], dtype=np.int32)
    y[:batch_size, 1] = 1
    y[batch_size:, 0] = 1

    # discriminatorの学習
    history = []
    for i in range(batch_size):
        history.append(discriminator_model.train_on_batch([en_input[i:i+2], ja_input_onehot[i:i+2]], y[i:i+2]))

    return history

# train generator
def train_generator(ja_seq_len, ja_vocab_size, x_train, batch_size=128):

    switch_trainable(discriminator_model, False)

    # batch_size数だけ、学習用の英語文をランダムに選択
    test_size = x_train.shape[0] - batch_size
    en_train, _, _, _ = train_test_split(x_train, x_train, test_size=test_size)

    # ja_seq_len数に増幅
    # en_train_xn = en_train
    # for _ in range(ja_seq_len-1):
    #     en_train_xn = np.append(en_train_xn, en_train, axis=0)

    # batch_size数だけ、日本語文初期値データを生成
    ja_initial_seq = onehot(initialize_seq(np.array([np.zeros(ja_seq_len, dtype=np.int32)])), ja_vocab_size)
    ja_generated = ja_initial_seq
    for _ in range(batch_size-1):
        ja_generated = np.append(ja_generated, ja_initial_seq, axis=0)
    # for i in range(batch_size):
    #     ja_output_seq = np.array([np.zeros(ja_seq_len)])
    #     for _ in range(ja_seq_len):
    #         ja_input_seq = initialize_seq(ja_output_seq)
    #         ja_generated = np.append(ja_generated, ja_input_seq, axis=0)
    #         ja_output_seq = predict(generator_model, en_train[i:i+1], ja_input_seq)
    # ja_generated = np.array(ja_generated[1:], dtype='int32')

    # 正解ラベルを用意
    y = np.zeros([batch_size, 2], dtype=np.int32)
    y[:, 1] = 1

    # ganを学習
    history = []
    for i in range(int(batch_size/2)):
        history.append(gan_model.train_on_batch([en_train[i:i+2], ja_generated[i:i+2], en_train[i:i+2]], y[i:i+2]))

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

    # loop (=epoch)
    for i in tqdm(range(step)):
        disc_history.extend(train_discriminator(ja_seq_len, ja_vocab_size, x_train, y_train, batch_size))
        gan_history.extend(train_generator(ja_seq_len, ja_vocab_size, x_train, batch_size))

        # 途中結果の確認
        if i % 10 == 9:
            translate(detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len, ja_vocab_size, test_count=2)
            # TODO: 途中結果の保存（generator/discriminatorの重み、学習結果(history)）

    return {'disc_history': disc_history, 'gan_history': gan_history}