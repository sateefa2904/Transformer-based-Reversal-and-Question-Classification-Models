#!/usr/bin/env python3
#Soli Ateefa
#1001924043
import numpy as np
import random, string
from reverse_base import train_enc_dec, get_enc_dec_results, word_accuracy
import os

def word_accuracy_simple(results, targets):
    """
    Compute word‐accuracy by lowercasing & stripping punctuation
    from the targets, then counting how many tokens match at each
    position (divided by the max length).
    """
    counts = 0
    totals = 0
    table = str.maketrans('', '', string.punctuation)
    for res, tgt in zip(results, targets):
        res_tokens = res.split()
        tgt_clean = tgt.lower().translate(table)
        tgt_tokens = tgt_clean.split()
        n_min = min(len(res_tokens), len(tgt_tokens))
        n_max = max(len(res_tokens), len(tgt_tokens))
        for i in range(n_min):
            if res_tokens[i] == tgt_tokens[i]:
                counts += 1
        totals += n_max
    return counts / totals

def load_data():
    base = os.path.dirname(__file__)           # path to your script
    ds   = os.path.join(base, 'reverse_dataset')

    # read train
    with open(os.path.join(ds, 'reverse_train.txt'), 'r') as f:
        train = [l.strip() for l in f if l.strip()]

    # read validation
    with open(os.path.join(ds, 'reverse_validation.txt'), 'r') as f:
        val   = [l.strip() for l in f if l.strip()]

    # read test
    test_inp, test_tgt = [], []
    with open(os.path.join(ds, 'reverse_test.txt'), 'r') as f:
        for line in f:
            src, tgt = line.strip().split('\t')
            test_inp.append(src)
            test_tgt.append(tgt)

    return train, val, test_inp, test_tgt

def main():
    train_sents, val_sents, test_sents, test_tgts = load_data()

    #test_sents = test_sents[:100]
    #test_tgts  = test_tgts[:100]

    scores = []
    epochs = 150

    for run in range(1, 11):
        print(f"Run {run}/10")
        model, src_vec, tgt_vec = train_enc_dec(train_sents, val_sents, epochs)
        preds = get_enc_dec_results(model, test_sents, src_vec, tgt_vec)
        acc   = word_accuracy_simple(preds, test_tgts) * 100
        print(f"  Accuracy: {acc:.2f}%")
        scores.append(acc)

    arr = np.array(scores)
    print("\nSummary over 10 runs:")
    print(f"Min Accuracy   : {arr.min():.2f}%")
    print(f"Max Accuracy   : {arr.max():.2f}%")
    print(f"Mean Accuracy  : {arr.mean():.2f}%")
    print(f"Median Accuracy: {np.median(arr):.2f}%")

if __name__=='__main__':
    main()
