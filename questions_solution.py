# Soli Ateefa
# 1001924043

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),])
        
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)


    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,})
        
        return config

def train_transformer(train_inputs, train_labels,
                      validation_inputs, validation_labels):
    max_tokens = 250
    max_length = 8
    vectorizer = TextVectorization(
        max_tokens = max_tokens,
        output_mode = "int",
        output_sequence_length = max_length
    )

    vectorizer.adapt(tf.constant(train_inputs))

    vocab_size = len(vectorizer.get_vocabulary())
    embed_dim  = 100 # change to 128.
    num_heads  = 3 # change to 4.
    dense_dim  = 32 # change to 64.

    inputs = keras.Input(shape=(), dtype=tf.string)
    x = vectorizer(inputs)
    x = PositionalEmbedding(max_length, vocab_size, embed_dim)(x)

    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
                                                               


    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x) 
    x = layers.Dense(20, activation="relu")(x) 
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = 0.0001), 
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    train_data = tf.constant(train_inputs)
    val_data   = tf.constant(validation_inputs)

    callbacks = [
        keras.callbacks.ModelCheckpoint("transformer.keras",
                                         save_best_only=True)]

    model.fit(
        train_data, train_labels,
        validation_data=(val_data, validation_labels),
        epochs=10, 
        batch_size=32,
        callbacks=callbacks
    )

    return model, vectorizer


def evaluate_transformer(model, text_vectorization,
                         test_inputs, test_labels):
    test_data = tf.constant(test_inputs)
    return model.evaluate(test_data, test_labels)[1]