"""
generate_gru.py
Generation RAPIDE par batch
Pour le modele GRU — compatible TF 1.14 / Python 3.6
"""

from __future__ import print_function
import sys
import numpy as np
import random
import os
import json

sys.stdout.flush()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(">>> STEP 1 : imports OK")
sys.stdout.flush()

import tensorflow as tf
print(">>> STEP 2 : tensorflow OK — version {}".format(tf.__version__))
sys.stdout.flush()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
print(">>> STEP 3 : keras OK")
sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement vocabulaire
# ─────────────────────────────────────────────────────────────────────────────

print(">>> STEP 4 : chargement vocabulary_gru.json ...")
sys.stdout.flush()

with open('vocabulary_gru.json', 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

chars       = vocab_data['chars']
char_to_int = vocab_data['char_to_int']
int_to_char = {int(k): v for k, v in vocab_data['int_to_char'].items()}
vocab_size  = len(chars)
seq_length  = 10

print(">>> STEP 4 OK : {} caracteres".format(vocab_size))
sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Reconstruction architecture GRU + chargement poids
# ─────────────────────────────────────────────────────────────────────────────

print(">>> STEP 5 : construction architecture GRU ...")
sys.stdout.flush()

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=seq_length))
model.add(GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.1))
model.add(GRU(128, dropout=0.3, recurrent_dropout=0.1))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)
print(">>> STEP 5 OK : architecture GRU construite")
sys.stdout.flush()

print(">>> STEP 6 : chargement poids ...")
sys.stdout.flush()

if os.path.exists('gru_password_generator.h5'):
    print(">>> gru_password_generator.h5 trouve — {} bytes".format(
        os.path.getsize('gru_password_generator.h5')))
    sys.stdout.flush()
    model.load_weights('gru_password_generator.h5')
    print(">>> STEP 6 OK : poids charges depuis gru_password_generator.h5")
elif os.path.exists('gru_best_model.h5'):
    print(">>> gru_best_model.h5 trouve — {} bytes".format(
        os.path.getsize('gru_best_model.h5')))
    sys.stdout.flush()
    model.load_weights('gru_best_model.h5')
    print(">>> STEP 6 OK : poids charges depuis gru_best_model.h5")
else:
    print(">>> ERREUR FATALE : aucun fichier GRU .h5 trouve !")
    sys.exit(1)

sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Chargement données train
# ─────────────────────────────────────────────────────────────────────────────

print(">>> STEP 7 : chargement train.txt ...")
sys.stdout.flush()

def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip().rstrip('\\') for line in f if line.strip()]
    return lines

train_lines = load_file('train.txt')

print(">>> STEP 7 OK : Train={}".format(len(train_lines)))
sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Test predict
# ─────────────────────────────────────────────────────────────────────────────

print(">>> STEP 8 : test predict() ...")
sys.stdout.flush()

x_test = np.zeros((1, seq_length), dtype=np.int32)
pred   = model.predict(x_test, verbose=0)
print(">>> STEP 8 OK : predict() fonctionne — shape={}".format(pred.shape))
sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Génération par BATCH
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE  = 256
TEMPERATURE = 1.0

def generate_batch(batch_size=256, max_length=20, temperature=1.0):
    sentences = []
    for _ in range(batch_size):
        seed = random.choice(train_lines)
        seed = seed.ljust(seq_length) if len(seed) < seq_length else seed[-seq_length:]
        sentences.append(list(seed))

    generated = ['' for _ in range(batch_size)]
    finished  = [False] * batch_size

    for _ in range(max_length):
        X_batch = np.zeros((batch_size, seq_length), dtype=np.int32)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X_batch[i, t] = char_to_int.get(char, 0)

        preds_batch = model.predict(X_batch, verbose=0)

        for i in range(batch_size):
            if finished[i]:
                continue
            preds   = preds_batch[i]
            clipped = np.clip(preds, 1e-10, 1.0)
            logits  = np.log(clipped) / temperature
            preds   = np.exp(logits)
            s       = np.sum(preds)
            preds   = preds / s if s > 1e-10 else np.ones_like(preds) / vocab_size
            next_index = np.random.choice(len(preds), p=preds)
            next_char  = int_to_char[next_index]
            if next_char == '\n':
                finished[i] = True
            else:
                generated[i] += next_char
                sentences[i]  = sentences[i][1:] + [next_char]

        if all(finished):
            break

    return generated

# ─────────────────────────────────────────────────────────────────────────────
# 6. Génération des runs
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = 'runs_gru'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_runs(n, temperature=1.0):
    passwords    = set()
    attempts     = 0
    max_attempts = n * 100

    while len(passwords) < n and attempts < max_attempts:
        batch = generate_batch(batch_size=BATCH_SIZE, max_length=20,
                               temperature=temperature)
        for pwd in batch:
            if pwd and len(pwd) >= 6:
                passwords.add(pwd)
        attempts += BATCH_SIZE

        if attempts % (BATCH_SIZE * 40) == 0:
            print("  ... {}/{} generes ({} tentatives)".format(
                len(passwords), n, attempts))
            sys.stdout.flush()

    return list(passwords)


print("\n" + "="*70)
print("GENERATION DES RUNS GRU — BATCH={} TEMP={}".format(BATCH_SIZE, TEMPERATURE))
print("="*70)
sys.stdout.flush()

for size, filename in [(10000, '10k.txt'), (100000, '100k.txt')]:
    filepath = os.path.join(OUTPUT_DIR, filename)
    print("\nGeneration de {:,} -> {} ...".format(size, filepath))
    sys.stdout.flush()

    passwords = generate_runs(n=size, temperature=TEMPERATURE)

    ttr     = len(set(passwords)) / len(passwords) if passwords else 0
    avg_len = np.mean([len(p) for p in passwords]) if passwords else 0

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(passwords))

    print("  {:,} ecrits dans '{}'".format(len(passwords), filepath))
    print("  TTR              : {:.4f}".format(ttr))
    print("  Longueur moyenne : {:.2f} chars".format(avg_len))
    sys.stdout.flush()

print("\n" + "="*70)
print("DONE — runs_gru/ generes avec succes")
print("="*70)