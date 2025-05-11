# reverse_solution_fixed.py
# Soli Ateefa
# 1001924043

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import random, string, re


def train_enc_dec(train_sentences, validation_sentences, epochs):
    # Determine max lengths
    max_src_len = max(len(s.split()) for s in train_sentences)
    max_tgt_len = max_src_len + 2  # [start] and [end]

    # Custom standardization to strip punctuation (keeping [start],[end])
    strip_chars = string.punctuation.replace("[", "").replace("]", "")
    def custom_standardization(txt):
        lower = tf.strings.lower(txt)
        return tf.strings.regex_replace(lower, f"[{re.escape(strip_chars)}]", "")

    # Vectorizers
    src_vectorizer = TextVectorization(
        output_mode="int",
        output_sequence_length=max_src_len,
        standardize=custom_standardization
    )
    src_vectorizer.adapt(train_sentences)

    tgt_texts = [f"[start] {s} [end]" for s in train_sentences]
    tgt_vectorizer = TextVectorization(
        output_mode="int",
        output_sequence_length=max_tgt_len,
        standardize=custom_standardization
    )
    tgt_vectorizer.adapt(tgt_texts)

    src_vocab = len(src_vectorizer.get_vocabulary())
    tgt_vocab = len(tgt_vectorizer.get_vocabulary())

    embed_dim = 128
    latent_dim = 128

    # Encoder
    enc_inputs = keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
    enc_emb = layers.Embedding(src_vocab, embed_dim, mask_zero=True)(enc_inputs)
    enc_outputs, state_h, state_c = layers.LSTM(
        latent_dim, return_sequences=True, return_state=True
    )(enc_emb)
    enc_states = [state_h, state_c]

    # Decoder
    dec_inputs = keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    dec_emb = layers.Embedding(tgt_vocab, embed_dim, mask_zero=True)(dec_inputs)
    dec_outputs, _, _ = layers.LSTM(
        latent_dim, return_sequences=True, return_state=True
    )(dec_emb, initial_state=enc_states)

    # Attention (additive)
    proj_enc = layers.TimeDistributed(layers.Dense(latent_dim))(enc_outputs)
    proj_dec = layers.TimeDistributed(layers.Dense(latent_dim))(dec_outputs)
    context = layers.AdditiveAttention()([proj_dec, proj_enc])

    # Concatenate and output
    concat = layers.Concatenate(axis=-1)([dec_outputs, context])
    outputs = layers.TimeDistributed(
        layers.Dense(tgt_vocab, activation="softmax"),
        name="decoder_outputs"
    )(concat)

    model = keras.Model([enc_inputs, dec_inputs], outputs)
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Prepare training data
    src_data, tgt_in, tgt_out = [], [], []
    for s in train_sentences:
        # random reverse augmentation
        src = s.split()[::-1] if random.random()<0.5 else s.split()
        src_data.append(" ".join(src))
        txt = f"[start] {s} [end]"
        seq = tgt_vectorizer([txt]).numpy()[0]
        tgt_in.append(seq[:-1])
        tgt_out.append(seq[1:])

    # Vectorize
    enc_data = src_vectorizer(np.array(src_data))
    dec_in = np.array(tgt_in)
    dec_out = np.expand_dims(np.array(tgt_out), -1)

    # Validation
    val_src = [ " ".join(s.split()[::-1]) if random.random()<0.5 else s for s in validation_sentences ]
    val_txt = [f"[start] {s} [end]" for s in validation_sentences]
    val_enc = src_vectorizer(np.array(val_src))
    val_seq = tgt_vectorizer(np.array(val_txt)).numpy()
    val_in = val_seq[:, :-1]
    val_out = np.expand_dims(val_seq[:, 1:], -1)

    model.fit(
        [enc_data, dec_in], dec_out,
        validation_data=([val_enc, val_in], val_out),
        batch_size=64, epochs=epochs
    )

    # Inference models
    enc_model = keras.Model(enc_inputs, enc_states)

    dec_state_h = keras.Input(shape=(latent_dim,))
    dec_state_c = keras.Input(shape=(latent_dim,))
    dec_state = [dec_state_h, dec_state_c]
    single = keras.Input(shape=(1,), dtype="int32")
    emb = dec_emb_layer = model.get_layer(index=3)(single)
    out, h2, c2 = model.get_layer("lstm")(emb, initial_state=dec_state)
    probs = model.get_layer("decoder_outputs")(layers.Concatenate(axis=-1)([out, context]))
    dec_model = keras.Model([single] + dec_state, [probs, h2, c2])

    model.encoder_model = enc_model
    model.decoder_model = dec_model
    return model, src_vectorizer, tgt_vectorizer


def get_enc_dec_results(model, test_sentences, src_vec, tgt_vec):
    vocab = tgt_vec.get_vocabulary()
    start_id = vocab.index("[start]")
    end_id   = vocab.index("[end]")
    max_len  = tgt_vec._output_sequence_length
    results = []
    for txt in test_sentences:
        states = model.encoder_model(src_vec([txt]))
        target = np.array([[start_id]])
        decoded = []
        for _ in range(max_len):
            proba, h, c = model.decoder_model([target] + states)
            choice = np.argmax(proba[0, -1, :])
            if choice==end_id: break
            decoded.append(vocab[choice])
            target = np.array([[choice]])
            states = [h, c]
        results.append(" ".join(decoded))
    return results


def train_best_model(train_sentences, validation_sentences):
    # Binary classifier for detecting reversed
    inputs, labels = [], []
    for s in train_sentences + validation_sentences:
        clean = re.sub(r"[^\w\s]", "", s.lower())
        inputs.append(clean); labels.append(0)
        inputs.append(" ".join(clean.split()[::-1])); labels.append(1)
    vec = TextVectorization(max_tokens=5000, output_mode="int", output_sequence_length=15)
    vec.adapt(inputs)
    X = vec(np.array(inputs)); y = np.array(labels)
    inp = keras.Input(shape=(None,), dtype="int32")
    x = layers.Embedding(5000, 128)(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    m.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)
    return m, vec, None


def get_best_model_results(model, test_sentences, src_vec, tgt_vec):
    results = []
    for s in test_sentences:
        clean = re.sub(r"[^\w\s]", "", s.lower())
        p = model.predict(src_vec([clean]))[0,0]
        words = clean.split()[::-1] if p>0.5 else clean.split()
        results.append(" ".join(words))
    return results
