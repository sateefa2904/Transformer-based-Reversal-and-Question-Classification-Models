import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

from questions_common import *
from questions_solution import *
from questions_common import load_inputs, load_labels
from questions_solution import train_transformer, evaluate_transformer


#%%

train_inputs = load_inputs("questions_dataset/questions_train.txt")
train_labels = load_labels("questions_dataset/questions_train_labels.txt")
validation_inputs = load_inputs("questions_dataset/questions_validation.txt")
validation_labels = load_labels("questions_dataset/questions_validation_labels.txt")
test_inputs = load_inputs("questions_dataset/questions_test.txt")
test_labels = load_labels("questions_dataset/questions_test_labels.txt")

print("read %d training sentences" % (len(train_inputs)))
print("read %d validation sentences" % (len(validation_inputs)))
print("read %d test examples" % (len(test_inputs)))

#%%

(model, text_vectorization) = train_transformer(train_inputs, train_labels, 
                                                validation_inputs, validation_labels)


accuracy = evaluate_transformer(model, text_vectorization, 
                                test_inputs, test_labels)

print("accuracy = %.2f%%" % (accuracy * 100))

