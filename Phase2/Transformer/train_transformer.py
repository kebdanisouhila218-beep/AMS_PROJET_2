# -*- coding: utf-8 -*-
"""
train_transformer.py

Transformer character-level pour generation de mots de passe.
Compatible : TF 1.14 / Python 3.6 / Tesla P100 / cuDNN 7

Architecture :
  Embedding + PositionalEncoding
  -> N x TransformerBlock (MultiHeadAttention + FFN + LayerNorm)
  -> Dense(vocab_size, softmax)

Avantage vs LSTM/GRU : meilleure capture des patterns globaux
=> meilleure couverture sur eval.txt
"""

from __future__ import print_function

import numpy as np
import os
import json
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Embedding,
    LayerNormalization, Add, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K

print("TF   :", tf.__version__)
print("Keras:", tf.keras.__version__)

# ─────────────────────────────────────────────────────────────────────────────
# GPU CONFIG — TF 1.14
# ─────────────────────────────────────────────────────────────────────────────
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

from tensorflow.python.client import device_lib
for d in device_lib.list_local_devices():
    if d.device_type == 'GPU':
        print("GPU detecte :", d.name)

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETRES
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LENGTH = 10     # longueur du contexte
D_MODEL    = 128    # dimension embedding + attention
NUM_HEADS  = 4      # tetes d attention (D_MODEL doit etre divisible par NUM_HEADS)
NUM_LAYERS = 2      # blocs Transformer empiles
D_FF       = 256    # dimension couche feed-forward interne
DROPOUT    = 0.1    # taux de dropout
BATCH_SIZE = 512
EPOCHS     = 20
LR         = 0.001

# ─────────────────────────────────────────────────────────────────────────────
# 0. Dossiers
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = 'runs_transformer'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print("Dossier '{}' cree.".format(OUTPUT_DIR))
else:
    print("Dossier '{}' deja existant.".format(OUTPUT_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement donnees
# ─────────────────────────────────────────────────────────────────────────────
def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

start = time.time()
train_lines = load_file('../../Data/train.txt')
text = '\n'.join(train_lines)
print("Train : {:,} mots de passe  ({:.2f}s)".format(len(train_lines), time.time() - start))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Vocabulaire
# ─────────────────────────────────────────────────────────────────────────────
chars       = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size  = len(chars)
print("Vocabulaire : {} caracteres".format(vocab_size))

vocab_data = {
    'chars'      : chars,
    'char_to_int': char_to_int,
    'int_to_char': {str(k): v for k, v in int_to_char.items()}
}
with open('vocabulary_transformer.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)
print("Vocabulaire sauvegarde -> vocabulary_transformer.json")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Sequences
# ─────────────────────────────────────────────────────────────────────────────
sentences  = []
next_chars = []

for i in range(0, len(text) - SEQ_LENGTH, 1):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_chars.append(text[i + SEQ_LENGTH])

print("Sequences : {:,}".format(len(sentences)))

X = np.zeros((len(sentences), SEQ_LENGTH), dtype=np.int32)
y = np.zeros((len(sentences), vocab_size),  dtype=np.float32)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_to_int.get(char, 0)
    y[i, char_to_int[next_chars[i]]] = 1.0

print("Shape X: {}  Shape y: {}".format(X.shape, y.shape))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Architecture Transformer
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(tf.keras.layers.Layer):
    """Encodage positionnel sinusoidal — ajoute la notion d ordre au Transformer."""
    def __init__(self, seq_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        # Precalcul de la matrice PE (seq_length, d_model)
        PE = np.zeros((seq_length, d_model), dtype=np.float32)
        for pos in range(seq_length):
            for i in range(0, d_model, 2):
                angle = pos / np.power(10000, (2 * i) / float(d_model))
                PE[pos, i]     = np.sin(angle)
                if i + 1 < d_model:
                    PE[pos, i+1] = np.cos(angle)
        # shape (1, seq_length, d_model) pour broadcast sur le batch
        self.PE = tf.constant(PE[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self.PE


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-Head Self-Attention compatible TF 1.14."""
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth     = d_model // num_heads
        self.d_model   = d_model

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.wo = Dense(d_model)

    def split_heads(self, x, batch_size):
        # (batch, seq, d_model) -> (batch, num_heads, seq, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]

        Q = self.split_heads(self.wq(x), batch_size)
        K = self.split_heads(self.wk(x), batch_size)
        V = self.split_heads(self.wv(x), batch_size)

        # Scaled dot-product attention
        d_k    = tf.cast(self.depth, tf.float32)
        scores  = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
        weights = tf.nn.softmax(scores, axis=-1)
        attn    = tf.matmul(weights, V)

        # Reassemble : (batch, num_heads, seq, depth) -> (batch, seq, d_model)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn, (batch_size, -1, self.d_model))

        return self.wo(attn)


class TransformerBlock(tf.keras.layers.Layer):
    """Un bloc Transformer : MHA + residuel + LayerNorm + FFN + residuel + LayerNorm."""
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.mha  = MultiHeadAttention(d_model, num_heads)
        self.ffn1 = Dense(d_ff,    activation='relu')
        self.ffn2 = Dense(d_model)
        self.ln1  = LayerNormalization(epsilon=1e-6)
        self.ln2  = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        # Bloc 1 : attention + residuel
        attn_out = self.mha(x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.ln1(x + attn_out)

        # Bloc 2 : feed-forward + residuel
        ffn_out = self.ffn2(self.ffn1(x))
        ffn_out = self.drop2(ffn_out, training=training)
        x = self.ln2(x + ffn_out)

        return x


class PasswordTransformer(tf.keras.Model):
    """Modele Transformer complet pour generation de mots de passe."""
    def __init__(self, vocab_size, seq_length, d_model, num_heads,
                 num_layers, d_ff, dropout_rate, **kwargs):
        super(PasswordTransformer, self).__init__(**kwargs)

        self.embedding = Embedding(vocab_size, d_model, input_length=seq_length)
        self.pos_enc   = PositionalEncoding(seq_length, d_model)
        self.dropout   = Dropout(dropout_rate)

        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.flatten = Flatten()
        self.dense   = Dense(vocab_size, activation='softmax')

    def call(self, x, training=False):
        x = self.embedding(x)           # (batch, seq, d_model)
        x = self.pos_enc(x)             # + encodage positionnel
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.flatten(x)             # (batch, seq * d_model)
        return self.dense(x)            # (batch, vocab_size)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Construction et compilation
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== CREATION DU MODELE TRANSFORMER ===")

model = PasswordTransformer(
    vocab_size   = vocab_size,
    seq_length   = SEQ_LENGTH,
    d_model      = D_MODEL,
    num_heads    = NUM_HEADS,
    num_layers   = NUM_LAYERS,
    d_ff         = D_FF,
    dropout_rate = DROPOUT
)

# Build explicite necessaire pour TF 1.14 avec sous-classes
model.build(input_shape=(None, SEQ_LENGTH))

model.compile(
    loss      = 'categorical_crossentropy',
    optimizer = Adam(lr=LR),
    metrics   = ['accuracy']
)

model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Callback
# ─────────────────────────────────────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    filepath      = 'transformer_best_model.h5',
    monitor       = 'val_loss',
    save_best_only= True,
    verbose       = 1
)

early_stop = EarlyStopping(
    monitor             = 'val_loss',
    patience            = 3,
    restore_best_weights= True,
    verbose             = 1
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Entrainement
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== DEBUT ENTRAINEMENT TRANSFORMER ===")
start_train = time.time()

history = model.fit(
    X, y,
    batch_size       = BATCH_SIZE,
    epochs           = EPOCHS,
    validation_split = 0.3,
    callbacks        = [early_stop, checkpoint],
    verbose          = 1
)

print("\nTemps total entrainement: {:.2f}s".format(time.time() - start_train))

# ─────────────────────────────────────────────────────────────────────────────
# 8. Metriques finales
# ─────────────────────────────────────────────────────────────────────────────
final_epoch    = len(history.history['loss'])
final_loss     = history.history['loss'][-1]
final_acc      = history.history['acc'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_acc  = history.history['val_acc'][-1]
final_ppl      = 2 ** final_loss
final_val_ppl  = 2 ** final_val_loss

print("\n" + "=" * 70)
print("METRIQUES FINALES — TRANSFORMER")
print("=" * 70)
print("Epochs            :", final_epoch)
print("Loss (train)      : {:.4f}".format(final_loss))
print("Accuracy (train)  : {:.4f}".format(final_acc))
print("Loss (val)        : {:.4f}".format(final_val_loss))
print("Accuracy (val)    : {:.4f}".format(final_val_acc))
print("Perplexite train  : {:.2f}".format(final_ppl))
print("Perplexite val    : {:.2f}".format(final_val_ppl))
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Sauvegarde finale
# ─────────────────────────────────────────────────────────────────────────────
model.save_weights("transformer_password_generator.h5")
print("\nPoids finaux sauvegardes    : transformer_password_generator.h5")
print("Meilleur modele sauvegarde  : transformer_best_model.h5")
print("Vocabulaire sauvegarde      : vocabulary_transformer.json")
print("\n=== ENTRAINEMENT TRANSFORMER TERMINE ===")