# 00_util

import numpy as np
import pickle
import csv
from nltk.translate.bleu_score import sentence_bleu

# detoken
def detoken(detokenizer, seqs, join_char=True):
    results = []
    for seq in seqs:
        detokened_seq = []
        for i in seq:
            if i == 0 or i == 1 or i == 2: # 0: padding, 1: bos, 2: eos
                continue
            detokened_seq.append(detokenizer[i])
        if join_char:
            results.append(' '.join(detokened_seq))
        else:
            results.append(detokened_seq)
    return results

# initialize_seq
def initialize_seq(ja_seqs):

    results = np.array([np.zeros(ja_seqs.shape[1], dtype=np.int32)])
    for i in range(ja_seqs.shape[0]):
        # 1文字目bos、以降はひとつスライド
        results = np.concatenate((results, np.array([np.append(np.array([1]), ja_seqs[i][:-1])])))

    return np.array(results[1:], dtype=np.int32)

# predict
def predict(en_input_seqs, ja_input_seqs):

    outputs = np.zeros((1, ja_seq_len), dtype=np.int32)
    for i in range(int(en_input_seqs.shape[0]/1000)): # for Memory Error
        from_i = 1000*i
        to_i = 1000*(i+1)
        outputs = np.concatenate((outputs, np.argmax(generator_model.predict([en_input_seqs[from_i:to_i], ja_input_seqs[from_i:to_i]]), axis=-1)))
    if en_input_seqs.shape[0] % 1000 != 0:
        from_i = int(en_input_seqs.shape[0]/1000) * 1000
        to_i = en_input_seqs.shape[0]
        outputs = np.concatenate((outputs, np.argmax(generator_model.predict([en_input_seqs[from_i:to_i], ja_input_seqs[from_i:to_i]]), axis=-1)))

    return outputs[1:]

# predict_all
def predict_all(en_input_seqs, ja_seq_len):

    ja_output_seqs = np.zeros((en_input_seqs.shape[0], ja_seq_len), dtype=np.int32)
    for _ in range(ja_seq_len):
        ja_input_seqs = initialize_seq(ja_output_seqs)
        ja_output_seqs = predict(en_input_seqs, ja_input_seqs)

    return ja_output_seqs

# translate sample
def translate_sample(detokenizer_en, detokenizer_ja, en_seqs, ja_seqs, ja_seq_len, test_count=5):
    en_trans_seqs = detoken(detokenizer_en, en_seqs[:test_count])
    ja_pred_seqs = detoken(detokenizer_ja, predict_all(en_seqs[:test_count], ja_seq_len))
    ja_trans_seqs = detoken(detokenizer_ja, ja_seqs[:test_count])

    for i in range(test_count):
        print('EN(', i, '): ', en_trans_seqs[i])
        print('JA-predict(', i, '): ', ja_pred_seqs[i])
        print('JA-answer (', i, '): ', ja_trans_seqs[i][1:])

# save all prediction as CSV
#   return seqs and ja predict seqs without padding
def save_all_prediction(outdir, en_seqs, ja_seq, ja_seq_len):

    # 予測後の日本語文の取得
    ja_pred_seqs = predict_all(en_seqs, ja_seq_len)

    # padding除去
    ja_seqs_without_pad = []
    ja_pred_seqs_without_pad = []
    for i in range(len(ja_pred_seqs)):

        without_pad = []
        for c in ja_seq[i]:
            if c == 0 or c == 1 or c == 2: # 0: padding, 1: bos, 2: eos
                continue
            without_pad.append(c)

        ja_seqs_without_pad.append(without_pad)

        without_pad = []
        for c in ja_pred_seqs[i]:
            if c == 0 or c == 1 or c == 2: # 0: padding, 1: bos, 2: eos
                continue
            without_pad.append(c)
        ja_pred_seqs_without_pad.append(without_pad)

    # CSV保存
    with open(outdir + "/valid.csv", 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(ja_pred_seqs_without_pad)

    return ja_seqs_without_pad, ja_pred_seqs_without_pad

# score BLEU
def scoreBLEU(detokenizer_ja, predict_seq, ref_seq):

    bleu = 0.0
    for i in range(len(predict_seq)):
        p = [1] # bos
        p.extend(predict_seq[i])
        p.append(2) # eos

        r = [1] # bos
        r.extend(ref_seq[i])
        r.append(2) # eos

        ref = detoken(detokenizer_ja, [r], False)[0]
        pred = detoken(detokenizer_ja, [p], False)[0]

        bleu += sentence_bleu([ref], pred)

    return bleu / len(predict_seq)

# save model
def save_model(outdir, prefix):
    generator_model.save_weights(outdir+'/'+prefix+'_generator_model_weight.h5')
    discriminator_model.save_weights(outdir+'/'+prefix+'_discriminator_model_weight.h5')

# save history
def save_history(outdir, history):
    with open(outdir+"/history.pickle", mode='wb') as f:
        pickle.dump(history, f)
