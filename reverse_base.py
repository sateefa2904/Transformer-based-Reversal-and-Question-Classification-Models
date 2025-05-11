#Soli Ateefa
#1001924043

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import random
import string
import re

from reverse_common import *
from reverse_solution import *


#%%

train_sentences = load_strings("reverse_dataset/reverse_train.txt")
validation_sentences = load_strings("reverse_dataset/reverse_validation.txt")
(test_sources, test_targets) = load_pairs("reverse_dataset/reverse_test.txt")
                                                      
print("read %d training sentences" % (len(train_sentences)))
print("read %d validation sentences" % (len(validation_sentences)))
print("read %d test examples" % (len(test_sources)))

#%%

(model, source_vec_layer, target_vec_layer) = train_enc_dec(train_sentences, 
                                                            validation_sentences,
                                                            150)

#%%
number = len(test_sources) 
(small_test_sources, small_test_targets) = random_samples(test_sources, 
                                                          test_targets, 
                                                          number)
results = get_enc_dec_results(model, small_test_sources,
                              source_vec_layer, target_vec_layer)

wa = word_accuracy(results, small_test_targets)
print("Encoder-decoder word accuracy = %.2f%%" % (wa * 100))

#%%

(model, source_vec_layer, target_vec_layer) = train_best_model(train_sentences, 
                                                               validation_sentences)


number = len(test_sources) 
(small_test_sources, small_test_targets) = random_samples(test_sources, 
                                                          test_targets, 
                                                          number)

results = get_best_model_results(model, small_test_sources,
                                 source_vec_layer, target_vec_layer)

wa = word_accuracy(results, small_test_targets)
print("Best model word accuracy = %.2f%%" % (wa * 100))



