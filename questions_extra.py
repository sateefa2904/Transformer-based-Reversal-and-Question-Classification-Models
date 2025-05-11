# questions_extra.py
# Soli Ateefa
# 1001924043

import numpy as np
import os
import string
import random

from questions_base import train_transformer, evaluate_transformer
from questions_common import load_inputs, load_labels


def main():
    base = os.path.dirname(__file__)
    ds = os.path.join(base, 'questions_dataset')

    # Load data
    train_inputs = load_inputs(os.path.join(ds, 'questions_train.txt'))
    train_labels = load_labels(os.path.join(ds, 'questions_train_labels.txt'))
    val_inputs = load_inputs(os.path.join(ds, 'questions_validation.txt'))
    val_labels = load_labels(os.path.join(ds, 'questions_validation_labels.txt'))
    test_inputs = load_inputs(os.path.join(ds, 'questions_test.txt'))
    test_labels = load_labels(os.path.join(ds, 'questions_test_labels.txt'))

    scores = []

    for run in range(1, 11):
        print(f"Run {run}/10")
        model, vectorizer = train_transformer(train_inputs, train_labels, val_inputs, val_labels)
        acc = evaluate_transformer(model, vectorizer, test_inputs, test_labels) * 100
        print(f"  Accuracy: {acc:.2f}%")
        scores.append(acc)

    arr = np.array(scores)
    print("\nSummary over 10 runs:")
    print(f"Min Accuracy   : {arr.min():.2f}%")
    print(f"Max Accuracy   : {arr.max():.2f}%")
    print(f"Mean Accuracy  : {arr.mean():.2f}%")
    print(f"Median Accuracy: {np.median(arr):.2f}%")


if __name__ == "__main__":
    main()
