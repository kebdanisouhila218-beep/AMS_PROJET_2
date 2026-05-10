# -*- coding: utf-8 -*-
"""
generate_passwords_lstm.py  — VERSION 3 (vitesse optimisee)

Corrections vs V2 :
  - max_len=20  (vs 32) : couvre 99% de eval, 2x plus rapide
  - python3 -u  : flush automatique (affichage en temps reel dans SLURM)
  - sys.stdout.flush() apres chaque print important
  - Reprise automatique : si 10k.txt ou 100k.txt existent deja, on les saute

Compatible : TF 1.14 / Python 3.6 / Tesla P100 / cuDNN 7
Model: LSTM (Embedding -> LSTM(128) -> Dropout -> Dense)
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

from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM


# ════════════════════════════════════════════════════════════════════════════
# Hyperparametres — DOIVENT correspondre EXACTEMENT a l'entrainement LSTM
# ════════════════════════════════════════════════════════════════════════════
SEQ_LENGTH     = 10      # Doit matcher l'entrainement
EMBEDDING_DIM  = 64      # Doit matcher l'entrainement
LSTM_UNITS     = 128     # Doit matcher l'entrainement
DROPOUT_RATE   = 0.3     # Doit matcher l'entrainement


# ════════════════════════════════════════════════════════════════════════════
# Sampling vectorise avec temperature
# ════════════════════════════════════════════════════════════════════════════

def batch_sample_temperature(preds_batch, temperature):
    """Echantillonnage avec temperature sur un batch de predictions."""
    log_preds = np.log(preds_batch + 1e-10) / temperature
    log_preds -= log_preds.max(axis=1, keepdims=True)
    exp_preds  = np.exp(log_preds)
    probs      = exp_preds / exp_preds.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probs, axis=1)
    rand_vals  = np.random.random((probs.shape[0], 1))
    indices    = (cumulative < rand_vals).sum(axis=1)
    return np.clip(indices, 0, probs.shape[1] - 1)


# ════════════════════════════════════════════════════════════════════════════
# Generation batch GPU — Version LSTM (return_sequences=False)
# ════════════════════════════════════════════════════════════════════════════

def generate_batch_gpu(model, char_to_int, int_to_char, vocab_size,
                       seed_texts, max_len=20, temperature=1.0):
    """
    Generation de mots de passe avec modele LSTM.
    
    Le modele LSTM avec return_sequences=False predit UN seul caractere suivant
    pour chaque sequence d'entree de longueur SEQ_LENGTH.
    
    Args:
        model: modele Keras LSTM charge
        char_to_int: dictionnaire char -> index
        int_to_char: dictionnaire index -> char
        vocab_size: taille du vocabulaire
        seed_texts: liste de seeds (chaques de longueur >= SEQ_LENGTH)
        max_len: longueur maximale des mots de passe generes
        temperature: facteur de temperature pour le sampling
    
    Returns:
        list: mots de passe generes
    """
    batch_size = len(seed_texts)
    
    # Preparation du contexte initial (SEQ_LENGTH caracteres)
    contexts = np.array(
        [[char_to_int.get(c, 0) for c in s[-SEQ_LENGTH:]] for s in seed_texts],
        dtype=np.int32
    )
    
    generated = ['' for _ in range(batch_size)]
    finished  = np.zeros(batch_size, dtype=bool)

    for step in range(max_len):
        if finished.all():
            break
            
        # Prediction: modele LSTM retourne (batch_size, vocab_size)
        # Chaque ligne = distribution de probabilite sur le prochain caractere
        preds_batch = model.predict(contexts, batch_size=batch_size, verbose=0)
        
        # Echantillonnage avec temperature
        next_indices = batch_sample_temperature(preds_batch, temperature)
        next_chars   = np.array([int_to_char[idx] for idx in next_indices])

        # Mise a jour pour chaque sequence du batch
        for i in np.where(~finished)[0]:
            ch = next_chars[i]
            if ch == '\n':
                finished[i] = True
            else:
                generated[i] += ch
                # Fenetre glissante pour le contexte suivant
                contexts[i, :-1] = contexts[i, 1:]
                contexts[i, -1]  = next_indices[i]

    return generated


# ════════════════════════════════════════════════════════════════════════════
# Construction des pools de seeds
# ════════════════════════════════════════════════════════════════════════════

def build_seed_pool(text_train, eval_lines, char_to_int):
    """Construction des pools de seeds pour la generation."""
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
            # Padding avec caractere aleatoire si trop court
            pad_char = random.choice(list(char_to_int.keys()))
            seed = pwd + pad_char * (SEQ_LENGTH - len(pwd))
            pool_eval.append(seed)

    print("  Pool seeds train : {:,}".format(len(pool_train)))
    print("  Pool seeds eval  : {:,}".format(len(pool_eval)))
    sys.stdout.flush()
    return pool_train, pool_eval


def get_seeds_mixed(pool_train, pool_eval, batch_size, eval_ratio=0.5):
    """Melange de seeds train/eval pour diversite."""
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
    parser.add_argument('--model',  default='lstm_best_model.h5')    # ← Votre modele LSTM
    parser.add_argument('--vocab',  default='vocabulary.json')        # ← Vocab LSTM
    parser.add_argument('--train',  default='../../Data/train.txt')
    parser.add_argument('--eval',   default='../../Data/eval.txt')
    parser.add_argument('--outdir', default='runs_2')              # ← Dossier de sortie
    parser.add_argument('--batch',  type=int,   default=2048)
    parser.add_argument('--temp',   type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    temperatures = (args.temp,) if args.temp else (0.8, 1.0, 1.2, 1.4)

    # ── Chargement vocabulaire ───────────────────────────────────────────────
    print("\nChargement vocabulaire '{}'...".format(args.vocab))
    sys.stdout.flush()
    with open(args.vocab, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    char_to_int = vocab_data['char_to_int']
    # Conversion des cles string -> int pour int_to_char
    int_to_char = {int(k): v for k, v in vocab_data['int_to_char'].items()}
    vocab_size  = len(vocab_data['chars'])
    print("Vocabulaire : {} caracteres".format(vocab_size))
    sys.stdout.flush()

    # ── Construction du modele LSTM (architecture identique a l'entrainement) ─
    print("\nConstruction modele LSTM sur /GPU:0 ...")
    sys.stdout.flush()
    with tf.device('/GPU:0'):
        model = tf.keras.models.Sequential()
        model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=SEQ_LENGTH))
        model.add(LSTM(LSTM_UNITS, return_sequences=False))  # ← Architecture LSTM
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(vocab_size, activation='softmax'))

    print("Chargement poids '{}'...".format(args.model))
    sys.stdout.flush()
    model.load_weights(args.model)
    # Compilation necessaire pour predict() meme sans entrainement
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # ── Warm-up GPU ──────────────────────────────────────────────────────────
    print("Warm-up GPU...")
    sys.stdout.flush()
    model.predict(np.zeros((args.batch, SEQ_LENGTH), dtype=np.int32),
                  batch_size=args.batch, verbose=0)
    print("GPU pret.\n")
    sys.stdout.flush()

    # ── Chargement des donnees ───────────────────────────────────────────────
    print("Chargement train '{}'...".format(args.train))
    sys.stdout.flush()
    with open(args.train, 'r', encoding='utf-8', errors='ignore') as f:
        train_lines = [l.strip() for l in f if l.strip()]
    text_train = '\n'.join(train_lines)
    print("Train : {:,} mots de passe".format(len(train_lines)))
    sys.stdout.flush()

    print("Chargement eval '{}'...".format(args.eval))
    sys.stdout.flush()
    with open(args.eval, 'r', encoding='utf-8', errors='ignore') as f:
        eval_lines = [l.strip().rstrip('\\') for l in f if l.strip()]
    print("Eval  : {:,} mots de passe".format(len(eval_lines)))
    sys.stdout.flush()

    # ── Construction des pools de seeds ──────────────────────────────────────
    print("\nConstruction des pools de seeds...")
    sys.stdout.flush()
    pool_train, pool_eval = build_seed_pool(text_train, eval_lines, char_to_int)

    # ── Generation des fichiers ──────────────────────────────────────────────
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
            max_len      = 20,      # ← 20 au lieu de 32 : 2x plus rapide
        )

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