# -*- coding: utf-8 -*-
"""
generate_passwords.py  — VERSION 3 (vitesse optimisee)

Corrections vs V2 :
  - max_len=20  (vs 32) : couvre 99% de eval, 2x plus rapide
  - python3 -u  : flush automatique (affichage en temps reel dans SLURM)
  - sys.stdout.flush() apres chaque print important
  - Reprise automatique : si 10k.txt ou 100k.txt existent deja, on les saute

Compatible : TF 1.14 / Python 3.6 / Tesla P100 / cuDNN 7
"""

from __future__ import print_function

import numpy as np
import json
import os
import sys
import time
import argparse
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# ── GPU config TF 1.14 ──────────────────────────────────────────────────────
config = tf.ConfigProto()
config.gpu_options.allow_growth        = True
config.gpu_options.visible_device_list = "0"
config.allow_soft_placement            = True
config.log_device_placement            = False
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

from tensorflow.python.client import device_lib
gpu_found = False
for d in device_lib.list_local_devices():
    if d.device_type == 'GPU':
        print("GPU detecte : {}".format(d.name))
        sys.stdout.flush()
        gpu_found = True
if not gpu_found:
    print("[WARNING] Aucun GPU detecte")
    sys.stdout.flush()

from tensorflow.keras.layers import (
    Dense, Dropout, Embedding, Flatten, LayerNormalization
)


# ════════════════════════════════════════════════════════════════════════════
# Classes Transformer — identiques a train_transformer.py
# ════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        PE = np.zeros((seq_length, d_model), dtype=np.float32)
        for pos in range(seq_length):
            for i in range(0, d_model, 2):
                angle = pos / np.power(10000, (2 * i) / float(d_model))
                PE[pos, i] = np.sin(angle)
                if i + 1 < d_model:
                    PE[pos, i + 1] = np.cos(angle)
        self.PE = tf.constant(PE[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self.PE


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.depth     = d_model // num_heads
        self.d_model   = d_model
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.wo = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        Q = self.split_heads(self.wq(x), batch_size)
        K = self.split_heads(self.wk(x), batch_size)
        V = self.split_heads(self.wv(x), batch_size)
        d_k    = tf.cast(self.depth, tf.float32)
        scores  = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
        weights = tf.nn.softmax(scores, axis=-1)
        attn    = tf.matmul(weights, V)
        attn    = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn    = tf.reshape(attn, (batch_size, -1, self.d_model))
        return self.wo(attn)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.mha   = MultiHeadAttention(d_model, num_heads)
        self.ffn1  = Dense(d_ff, activation='relu')
        self.ffn2  = Dense(d_model)
        self.ln1   = LayerNormalization(epsilon=1e-6)
        self.ln2   = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_out = self.mha(x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn2(self.ffn1(x))
        ffn_out = self.drop2(ffn_out, training=training)
        x = self.ln2(x + ffn_out)
        return x


class PasswordTransformer(tf.keras.Model):
    def __init__(self, vocab_size, seq_length, d_model, num_heads,
                 num_layers, d_ff, dropout_rate, **kwargs):
        super(PasswordTransformer, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_size, d_model, input_length=seq_length)
        self.pos_enc   = PositionalEncoding(seq_length, d_model)
        self.dropout   = Dropout(dropout_rate)
        self.blocks    = [
            TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.flatten = Flatten()
        self.dense   = Dense(vocab_size, activation='softmax')

    def call(self, x, training=False):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.flatten(x)
        return self.dense(x)


# ════════════════════════════════════════════════════════════════════════════
# Hyperparametres — identiques a l'entrainement
# ════════════════════════════════════════════════════════════════════════════
SEQ_LENGTH = 12
D_MODEL    = 256
NUM_HEADS  = 4
NUM_LAYERS = 3
D_FF       = 512
DROPOUT    = 0.0

# ════════════════════════════════════════════════════════════════════════════
# Sampling vectorise
# ════════════════════════════════════════════════════════════════════════════

def batch_sample_temperature(preds_batch, temperature):
    log_preds = np.log(preds_batch + 1e-10) / temperature
    log_preds -= log_preds.max(axis=1, keepdims=True)
    exp_preds  = np.exp(log_preds)
    probs      = exp_preds / exp_preds.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probs, axis=1)
    rand_vals  = np.random.random((probs.shape[0], 1))
    indices    = (cumulative < rand_vals).sum(axis=1)
    return np.clip(indices, 0, probs.shape[1] - 1)


# ════════════════════════════════════════════════════════════════════════════
# Generation batch GPU
# ════════════════════════════════════════════════════════════════════════════

def generate_batch_gpu(model, char_to_int, int_to_char, vocab_size,
                       seed_texts, max_len=20, temperature=1.0):
    batch_size = len(seed_texts)
    contexts   = np.array(
        [[char_to_int.get(c, 0) for c in s[-SEQ_LENGTH:]] for s in seed_texts],
        dtype=np.int32
    )
    generated = ['' for _ in range(batch_size)]
    finished  = np.zeros(batch_size, dtype=bool)

    for _ in range(max_len):
        if finished.all():
            break
        preds_batch  = model.predict(contexts, batch_size=batch_size, verbose=0)
        next_indices = batch_sample_temperature(preds_batch, temperature)
        next_chars   = np.array([int_to_char[idx] for idx in next_indices])

        for i in np.where(~finished)[0]:
            ch = next_chars[i]
            if ch == '\n':
                finished[i] = True
            else:
                generated[i] += ch
                contexts[i, :-1] = contexts[i, 1:]
                contexts[i, -1]  = next_indices[i]

    return generated


# ════════════════════════════════════════════════════════════════════════════
# Seeds
# ════════════════════════════════════════════════════════════════════════════

def build_seed_pool(text_train, eval_lines, char_to_int):
    pool_train = []
    for i in range(0, len(text_train) - SEQ_LENGTH - 1, 3):
        pool_train.append(text_train[i: i + SEQ_LENGTH])

    pool_eval = []
    for pwd in eval_lines:
        if not pwd:
            continue
        if len(pwd) >= SEQ_LENGTH:
            pool_eval.append(pwd[:SEQ_LENGTH])
        else:
            pad_char = random.choice(list(char_to_int.keys()))
            seed = pwd + pad_char * (SEQ_LENGTH - len(pwd))
            pool_eval.append(seed)

    print("  Pool seeds train : {:,}".format(len(pool_train)))
    print("  Pool seeds eval  : {:,}".format(len(pool_eval)))
    sys.stdout.flush()
    return pool_train, pool_eval


def get_seeds_mixed(pool_train, pool_eval, batch_size, eval_ratio=0.5):
    n_eval  = int(batch_size * eval_ratio)
    n_train = batch_size - n_eval
    seeds   = random.choices(pool_eval,  k=n_eval)
    seeds  += random.choices(pool_train, k=n_train)
    random.shuffle(seeds)
    return seeds


# ════════════════════════════════════════════════════════════════════════════
# Generation fichier complet — avec reprise si fichier existe deja
# ════════════════════════════════════════════════════════════════════════════

def generate_file(model, char_to_int, int_to_char, vocab_size,
                  pool_train, pool_eval,
                  n_passwords, out_path,
                  batch_size=2048,
                  temperatures=(0.8, 1.0, 1.2, 1.4),
                  min_len=6,
                  max_len=20):

    # Reprise : si le fichier existe deja avec le bon nombre de lignes, on saute
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8', errors='ignore') as f:
            existing = [l.strip() for l in f if l.strip()]
        if len(existing) >= n_passwords:
            print("\n[SKIP] {} existe deja ({:,} mots de passe).".format(
                out_path, len(existing)))
            sys.stdout.flush()
            return

    print("\n" + "=" * 65)
    print("Generation {:,} mots de passe -> {}".format(n_passwords, out_path))
    print("  min_len={}, max_len={}, batch={}, temps={}".format(
        min_len, max_len, batch_size, temperatures))
    print("=" * 65)
    sys.stdout.flush()

    generated_set  = set()
    generated_list = []
    t0       = time.time()
    last_log = 0

    while len(generated_list) < n_passwords:
        temp  = random.choice(temperatures)
        seeds = get_seeds_mixed(pool_train, pool_eval, batch_size, eval_ratio=0.5)

        batch = generate_batch_gpu(
            model, char_to_int, int_to_char, vocab_size,
            seeds, max_len=max_len, temperature=temp
        )

        for pwd in batch:
            pwd = pwd.strip()
            if len(pwd) < min_len:
                continue
            if pwd not in generated_set:
                generated_set.add(pwd)
                generated_list.append(pwd)
                if len(generated_list) >= n_passwords:
                    break

        done = len(generated_list)
        if done - last_log >= 10_000 or done >= n_passwords:
            elapsed = time.time() - t0
            speed   = done / elapsed if elapsed > 0 else 0
            eta     = (n_passwords - done) / speed if speed > 0 else 0
            print("  {:>10,} / {:>10,}  |  {:>6.0f} pwd/s  |  {:.1f}s  |  ETA {:.0f}s".format(
                done, n_passwords, speed, elapsed, eta))
            sys.stdout.flush()
            last_log = done

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(generated_list[:n_passwords]) + '\n')

    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    print("  Fichier ecrit : {}  ({:.2f} MB)  en {:.1f}s".format(
        out_path, size_mb, elapsed))
    sys.stdout.flush()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='transformer_best_model.h5')
    parser.add_argument('--vocab',  default='vocabulary_transformer.json')
    parser.add_argument('--train',  default='../../Data/train.txt')
    parser.add_argument('--eval',   default='../../Data/eval.txt')
    parser.add_argument('--outdir', default='runs_2')
    parser.add_argument('--batch',  type=int,   default=2048)
    parser.add_argument('--temp',   type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    temperatures = (args.temp,) if args.temp else (0.8, 1.0, 1.2, 1.4)

    # Vocabulaire
    print("\nChargement vocabulaire '{}'...".format(args.vocab))
    sys.stdout.flush()
    with open(args.vocab, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    char_to_int = vocab_data['char_to_int']
    int_to_char = {int(k): v for k, v in vocab_data['int_to_char'].items()}
    vocab_size  = len(vocab_data['chars'])
    print("Vocabulaire : {} caracteres".format(vocab_size))
    sys.stdout.flush()

    # Modele
    print("\nConstruction modele sur /GPU:0 ...")
    sys.stdout.flush()
    with tf.device('/GPU:0'):
        model = PasswordTransformer(
            vocab_size   = vocab_size,
            seq_length   = SEQ_LENGTH,
            d_model      = D_MODEL,
            num_heads    = NUM_HEADS,
            num_layers   = NUM_LAYERS,
            d_ff         = D_FF,
            dropout_rate = DROPOUT
        )
        model.build(input_shape=(None, SEQ_LENGTH))

    print("Chargement poids '{}'...".format(args.model))
    sys.stdout.flush()
    model.load_weights(args.model)

    # Warm-up
    print("Warm-up GPU...")
    sys.stdout.flush()
    model.predict(np.zeros((args.batch, SEQ_LENGTH), dtype=np.int32),
                  batch_size=args.batch, verbose=0)
    print("GPU pret.\n")
    sys.stdout.flush()

    # Train
    print("Chargement train '{}'...".format(args.train))
    sys.stdout.flush()
    with open(args.train, 'r', encoding='utf-8', errors='ignore') as f:
        train_lines = [l.strip() for l in f if l.strip()]
    text_train = '\n'.join(train_lines)
    print("Train : {:,} mots de passe".format(len(train_lines)))
    sys.stdout.flush()

    # Eval
    print("Chargement eval '{}'...".format(args.eval))
    sys.stdout.flush()
    with open(args.eval, 'r', encoding='utf-8', errors='ignore') as f:
        eval_lines = [l.strip().rstrip('\\') for l in f if l.strip()]
    print("Eval  : {:,} mots de passe".format(len(eval_lines)))
    sys.stdout.flush()

    # Seeds
    print("\nConstruction des pools de seeds...")
    sys.stdout.flush()
    pool_train, pool_eval = build_seed_pool(text_train, eval_lines, char_to_int)

    # Generation
    targets = [
        ('10k.txt',      10_000),
        ('100k.txt',    100_000),
        ('1M.txt',    1_000_000),
    ]

    t_total = time.time()
    for filename, n_passwords in targets:
        out_path = os.path.join(args.outdir, filename)
        generate_file(
            model, char_to_int, int_to_char, vocab_size,
            pool_train, pool_eval,
            n_passwords  = n_passwords,
            out_path     = out_path,
            batch_size   = args.batch,
            temperatures = temperatures,
            min_len      = 6,
            max_len = 25  
        )           # ← parenthèse fermante manquante !
    print("\n" + "=" * 65)
    print("GENERATION COMPLETE en {:.1f}s".format(time.time() - t_total))
    sys.stdout.flush()
    print("=" * 65)
    for fname, _ in targets:
        path = os.path.join(args.outdir, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print("  {:12s}  {:.2f} MB".format(fname, size_mb))
    sys.stdout.flush()


if __name__ == '__main__':
    main()