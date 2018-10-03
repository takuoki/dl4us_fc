# 00_util

import numpy as np

# detoken
def detoken(detokenizer, seqs):
    results = []
    for seq in seqs:
        detokened_seq = []
        for i in seq:
            if i == 0:
                break
            detokened_seq.append(detokenizer[i])
        results.append(' '.join(detokened_seq))
    return results

# translate
def translate(detokenizer_en, detokenizer_ja, x_valid, y_valid, ja_seq_len, test_count=5):
    en_seqs = detoken(detokenizer_en, x_valid[:test_count])
    ja_pred_seqs = detoken(detokenizer_ja, predict_all(x_valid[:test_count], ja_seq_len))
    ja_seqs = detoken(detokenizer_ja, y_valid[:test_count])

    for i in range(test_count):
        print('EN(', i, '): ', en_seqs[i])
        print('JA-predict(', i, '): ', ja_pred_seqs[i])
        print('JA-answer (', i, '): ', ja_seqs[i])

# TODO: 精度検証(BLUE?)
