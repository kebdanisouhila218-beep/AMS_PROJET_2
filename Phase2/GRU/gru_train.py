

from __future__ import print_function

import numpy as np
import random
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("TF   :", tf.__version__)
print("Keras:", tf.keras.__version__)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Dossier de sortie
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = 'runs_gru'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print("Dossier '{}' cree.".format(OUTPUT_DIR))
else:
    print("Dossier '{}' deja existant.".format(OUTPUT_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

train_lines = load_file('train.txt')

text        = '\n'.join(train_lines)

print("Train :", len(train_lines), "mots de passe")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prétraitement + Sauvegarde vocabulaire
# ─────────────────────────────────────────────────────────────────────────────

chars       = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size  = len(chars)
print("Vocabulaire :", vocab_size, "caracteres")

vocab_data = {
    'chars': chars,
    'char_to_int': char_to_int,
    'int_to_char': {str(k): v for k, v in int_to_char.items()}
}
with open('vocabulary_gru.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)
print("Vocabulaire sauvegarde -> vocabulary_gru.json")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Séquences
# ─────────────────────────────────────────────────────────────────────────────

seq_length = 10
step       = 1
sentences  = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

print("Sequences :", len(sentences))

X = np.zeros((len(sentences), seq_length), dtype=np.int32)
y = np.zeros((len(sentences), vocab_size),  dtype=np.float32)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_to_int.get(char, 0)
    y[i, char_to_int[next_chars[i]]] = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 4. Modèle GRU
#    Même architecture que LSTM mais avec GRU
#    GRU = moins de paramètres (~25% de moins) = plus rapide à entraîner
# ─────────────────────────────────────────────────────────────────────────────

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=seq_length))

model.add(GRU(128, return_sequences=True,
              dropout=0.3, recurrent_dropout=0.1))
model.add(GRU(128,
              dropout=0.3, recurrent_dropout=0.1))

model.add(Dense(vocab_size, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Callbacks
# ─────────────────────────────────────────────────────────────────────────────

checkpoint = ModelCheckpoint(
    filepath='gru_best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Entraînement
# ─────────────────────────────────────────────────────────────────────────────

history = model.fit(
    X, y,
    batch_size=256,
    epochs=50,
    validation_split=0.1,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Métriques finales
# ─────────────────────────────────────────────────────────────────────────────

final_epoch    = len(history.history['loss'])
final_loss     = history.history['loss'][-1]
final_acc      = history.history['acc'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_acc  = history.history['val_acc'][-1]
final_ppl      = 2 ** final_loss
final_val_ppl  = 2 ** final_val_loss

print("\n" + "="*70)
print("MÉTRIQUES FINALES — GRU")
print("="*70)
print("Epochs            :", final_epoch)
print("Loss (train)      : {:.4f}".format(final_loss))
print("Accuracy (train)  : {:.4f}".format(final_acc))
print("Loss (val)        : {:.4f}".format(final_val_loss))
print("Accuracy (val)    : {:.4f}".format(final_val_acc))
print("Perplexite train  : {:.2f}".format(final_ppl))
print("Perplexite val    : {:.2f}".format(final_val_ppl))
print("="*70)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Sauvegarde finale
# ─────────────────────────────────────────────────────────────────────────────

model.save("gru_password_generator.h5")
print("\nModele final sauvegarde    : gru_password_generator.h5")
print("Meilleur modele sauvegarde : gru_best_model.h5")
print("Vocabulaire sauvegarde     : vocabulary_gru.json")