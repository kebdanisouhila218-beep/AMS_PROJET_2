# -*- coding: utf-8 -*-
"""
evaluate_transformer.py

Evaluation des runs Transformer generees contre eval.txt
Inspire de evaluate_gru.py — adapte pour les 3 runs : 10k / 100k / 1M

Metriques :
  - Couverture eval set (hits / taille eval)
  - Entropie reelle (cible >= 40 bits)
  - Taux de doublons (cible < 5%)
  - Force du mot de passe (distribution)
  - TTR, longueur moyenne, bigrammes

Log affiche ET enregistre dans evaluate_transformer.log
"""

from __future__ import print_function

import sys
import os
import math
import re
from datetime import datetime
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# Logger — ecrit a la fois sur stdout et dans un fichier log
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE = 'evaluate_transformer_2.log'

class Logger:
    def __init__(self, filepath):
        self.console = sys.stdout
        self.logfile = open(filepath, 'a', encoding='utf-8')

    def write(self, message):
        self.console.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.console.flush()
        self.logfile.flush()

sys.stdout = Logger(LOG_FILE)

print("\n" + "=" * 70)
print("RUN DATE : {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
print("MODELE   : TRANSFORMER")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def load_file(filepath):
    """
    Charge un fichier de mots de passe.
    CORRECTION vs evaluate_gru.py :
      - encoding='utf-8' + errors='ignore' pour eviter les crash sur
        caracteres non-ASCII presents dans les runs Transformer
      - rstrip('\\') conserve (utile si des lignes finissent par backslash)
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip().rstrip('\\') for line in f if line.strip()]
    return lines


def mean(values):
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def compute_entropy(passwords):
    """
    Entropie reelle (bits) = entropie par caractere x longueur moyenne.
    Mesure la variete des caracteres utilises.
    """
    if not passwords:
        return 0.0
    all_chars = ''.join(passwords)
    if not all_chars:
        return 0.0
    freq  = Counter(all_chars)
    total = float(len(all_chars))
    entropy_per_char = -sum(
        (c / total) * math.log(c / total, 2)
        for c in freq.values()
    )
    avg_len = mean([len(p) for p in passwords])
    return entropy_per_char * avg_len


def compute_duplicate_rate(passwords):
    """Pourcentage de doublons dans la liste generee."""
    if not passwords:
        return 0.0
    total  = float(len(passwords))
    unique = float(len(set(passwords)))
    return (total - unique) / total * 100.0


def classify_strength(password):
    """Classifie un mot de passe selon sa complexite."""
    has_lower   = bool(re.search(r'[a-z]',       password))
    has_upper   = bool(re.search(r'[A-Z]',       password))
    has_digit   = bool(re.search(r'\d',          password))
    has_special = bool(re.search(r'[^a-zA-Z0-9]', password))
    types  = sum([has_lower, has_upper, has_digit, has_special])
    length = len(password)
    if length < 6 or types < 2:
        return 'Faible'
    if length >= 12 and types >= 3:
        return 'Fort'
    if length >= 8 and types >= 3:
        return 'Mixte'
    return 'Moyen'


def compute_strength_distribution(passwords):
    if not passwords:
        return {}
    dist  = Counter(classify_strength(p) for p in passwords)
    total = float(len(passwords))
    return {k: dist.get(k, 0) / total * 100.0
            for k in ['Faible', 'Moyen', 'Mixte', 'Fort']}


def compute_coverage(gen_set, eval_set):
    """% de mots de eval.txt retrouves dans les mots generes."""
    if not eval_set:
        return 0.0
    return float(len(gen_set & eval_set)) / float(len(eval_set)) * 100.0


def compute_bigram_coverage(eval_passwords, gen_passwords):
    """
    Couverture en bigrammes : mesure si les transitions de caracteres
    apprises par le Transformer sont bien restituees.
    """
    def bigrams(pwd):
        return set(pwd[i:i+2] for i in range(len(pwd) - 1))

    eval_bg = set()
    for p in eval_passwords:
        eval_bg |= bigrams(p)

    gen_bg = set()
    for p in gen_passwords:
        gen_bg |= bigrams(p)

    if not eval_bg:
        return 0.0, 0, 0
    covered = eval_bg & gen_bg
    return float(len(covered)) / float(len(eval_bg)) * 100.0, len(covered), len(eval_bg)


def security_status(value, target, mode='gte'):
    if mode == 'gte':
        return '[OK]' if value >= target else '[!!]'
    if mode == 'lte':
        return '[OK]' if value <= target else '[!!]'
    return '[??]'


def print_bar(label, pct, marker=''):
    bar = '#' * int(pct / 2)
    print("    {:6s} : {:5.1f}%  {}{}".format(label, pct, bar, marker))


# ─────────────────────────────────────────────────────────────────────────────
# Chargement eval.txt
# ─────────────────────────────────────────────────────────────────────────────

EVAL_PATH = '../../Data/eval.txt'

print("\n>>> Chargement {} ...".format(EVAL_PATH))
if not os.path.exists(EVAL_PATH):
    print("[ERREUR] Fichier introuvable : {}".format(EVAL_PATH))
    print("         Verifie le chemin ou copie eval.txt dans le dossier courant.")
    sys.exit(1)

eval_lines = load_file(EVAL_PATH)
eval_set   = set(eval_lines)
print(">>> eval.txt : {:,} mots de passe  ({:,} uniques)".format(
    len(eval_lines), len(eval_set)))


# ─────────────────────────────────────────────────────────────────────────────
# Runs a evaluer
# ─────────────────────────────────────────────────────────────────────────────

RUNS_DIR = 'runss_transformer'

run_files = [
    ('10k.txt',  '10k'),
    ('100k.txt', '100k'),
    ('1M.txt',   '1M'),      # AJOUT par rapport a evaluate_gru.py
]

print("\n" + "=" * 70)
print("EVALUATION TRANSFORMER vs EVAL.TXT")
print("=" * 70)

for filename, label in run_files:
    filepath = os.path.join(RUNS_DIR, filename)

    if not os.path.exists(filepath):
        print("\n[{}] FICHIER INTROUVABLE : {}".format(label, filepath))
        print("     Lance d'abord : python3 generate_passwords.py")
        continue

    print("\n[{}] Chargement {} ...".format(label, filepath))
    gen_lines = load_file(filepath)

    if not gen_lines:
        print("[{}] FICHIER VIDE — verifie la generation.".format(label))
        continue

    gen_set = set(gen_lines)

    # ── Metriques generales ──────────────────────────────────────────────────
    ttr      = float(len(gen_set)) / float(len(gen_lines))
    avg_len  = mean([len(p) for p in gen_lines])
    hits     = eval_set & gen_set
    coverage = compute_coverage(gen_set, eval_set)

    # ── Metriques de securite ────────────────────────────────────────────────
    entropy       = compute_entropy(gen_lines)
    dup_rate      = compute_duplicate_rate(gen_lines)
    strength_dist = compute_strength_distribution(gen_lines)

    # ── Couverture bigrammes ─────────────────────────────────────────────────
    bg_cov, bg_hit, bg_total = compute_bigram_coverage(eval_lines, gen_lines)

    # ── Affichage ────────────────────────────────────────────────────────────
    print("")
    print("  -- Metriques generales ------------------------------------------")
    print("  Mots de passe generes (total) : {:,}".format(len(gen_lines)))
    print("  Mots de passe uniques         : {:,}".format(len(gen_set)))
    print("  TTR                           : {:.4f}".format(ttr))
    print("  Longueur moyenne              : {:.2f} chars".format(avg_len))
    print("  Hits sur eval                 : {:,}".format(len(hits)))
    print("  Couverture eval               : {:.4f}%".format(coverage))
    print("  Couverture bigrammes          : {:.2f}%  ({:,} / {:,})".format(
        bg_cov, bg_hit, bg_total))

    print("")
    print("  -- Metriques de securite ----------------------------------------")
    print("  {:<6} Entropie reelle        : {:.2f} bits  (cible >= 40 bits)".format(
        security_status(entropy, 40, 'gte'), entropy))
    print("  {:<6} Taux de doublons       : {:.2f}%      (cible < 5%)".format(
        security_status(dup_rate, 5, 'lte'), dup_rate))

    print("")
    print("  Force des mots de passe (distribution) :")
    for level in ['Faible', 'Moyen', 'Mixte', 'Fort']:
        pct    = strength_dist.get(level, 0.0)
        marker = ' <-- CIBLE' if level == 'Mixte' else ''
        print_bar(level, pct, marker)

    mixte_pct = strength_dist.get('Mixte', 0.0) + strength_dist.get('Fort', 0.0)
    print("")
    print("  {:<6} Mots Mixte+Fort  : {:.1f}%  (cible : Mixte dominant)".format(
        '[OK]' if mixte_pct >= 30 else '[!!]', mixte_pct))
    print("  {:<6} Couverture eval  : {:.4f}%  (cible : Maximale)".format(
        '[OK]' if coverage > 0 else '[!!]', coverage))

    if hits:
        exemples = list(hits)[:10]
        print("")
        print("  Exemples de hits ({}) :".format(len(hits)))
        for ex in exemples:
            print("    - {}".format(ex))

    print("  " + "-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Fin
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DONE — evaluation Transformer terminee")
print("=" * 70)
print(">>> Log sauvegarde dans : {}".format(LOG_FILE))