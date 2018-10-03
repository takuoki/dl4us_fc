# 01_dataload

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

def load_data(root_dir, valid_size=1000):
    # load data
    data_dir = root_dir + 'data/'
    x_train = np.load(data_dir + 'x_train.npy')
    y_train = np.load(data_dir + 'y_train.npy')

    # load tokenizer
    tokenizer_en = np.load(data_dir + 'tokenizer_en.npy').item()
    tokenizer_ja = np.load(data_dir + 'tokenizer_ja.npy').item()
    detokenizer_en = dict(map(reversed, tokenizer_en.word_index.items()))
    detokenizer_ja = dict(map(reversed, tokenizer_ja.word_index.items()))

    # split data
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size, random_state=42)

    return (x_train, y_train, x_valid, y_valid, tokenizer_en, tokenizer_ja, detokenizer_en, detokenizer_ja)

def show_data(valid_size=500):
    final_dir = '/root/userspace/final/'
    x_train, y_train, x_valid, y_valid, _, _, detokenizer_en, detokenizer_ja = load_data(final_dir, valid_size)

    max_x = 0
    for i in range(x_train.shape[0]):
        if x_train[i].shape[0] > max_x:
            max_x = x_train[i].shape[0]
    for i in range(x_valid.shape[0]):
        if x_valid[i].shape[0] > max_x:
            max_x = x_valid[i].shape[0]

    max_y = 0
    for i in range(y_train.shape[0]):
        if y_train[i].shape[0] > max_y:
            max_y = y_train[i].shape[0]
    for i in range(y_valid.shape[0]):
        if y_valid[i].shape[0] > max_y:
            max_y = y_valid[i].shape[0]

    print('# count')
    print('x_train: ', x_train.shape[0])
    print('y_train: ', y_train.shape[0])
    print('x_valid: ', x_valid.shape[0])
    print('y_valid: ', y_valid.shape[0])
    print('max_length(x, y): (', max_x, ', ', max_y, ')')
    print()

    sample_data_cnt = 10
    print('# data sample (cnt=', sample_data_cnt, ')')

    for text_no in range(sample_data_cnt):
        x_train_text = [detokenizer_en[i] for i in x_train[text_no].tolist() if i !=0]
        y_train_text = [detokenizer_ja[i] for i in y_train[text_no].tolist() if i !=0]
        
        print('x_train(' + str(text_no) + '): ', x_train_text[1:-1])
        print('y_train(' + str(text_no) + '): ', y_train_text[1:-1])

show_data(1000)
