# -*- coding: utf-8 -*-
"""
evaluate_gru.py

Evaluation des runs GRU generees contre eval.txt
Inclut les metriques de securite (section 4.3.4) :
  - Entropie reelle (cible >= 40 bits)
  - Taux de doublons (cible < 5%)
  - Force du mot de passe (Mixte)
  - Couverture eval set (Maximale)

A lancer APRES generate_gru.py
Les logs sont affiches ET enregistres dans evaluate_gru.log
"""

from __future__ import print_function

import sys
import os
import math
import re
from datetime import datetime
from collections import Counter

LOG_FILE = 'evaluate_gru.log'

class Logger:
    def __init__(self, filepath):
        self.console = sys.stdout
        self.logfile = open(filepath, 'a')

    def write(self, message):
        self.console.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.console.flush()
        self.logfile.flush()

sys.stdout = Logger(LOG_FILE)

print("\n" + "="*70)
print("RUN DATE : {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
print("MODELE   : GRU")
print("="*70)

def load_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip().rstrip('\\') for line in f if line.strip()]
    return lines

def mean(values):
    if not values:
        return 0.0
    return sum(values) / float(len(values))

def compute_entropy(passwords):
    if not passwords:
        return 0.0
    all_chars = ''.join(passwords)
    freq = Counter(all_chars)
    total = float(len(all_chars))
    entropy_per_char = -sum((c / total) * math.log(c / total, 2) for c in freq.values())
    avg_len = mean([len(p) for p in passwords])
    return entropy_per_char * avg_len

def compute_duplicate_rate(passwords):
    if not passwords:
        return 0.0
    total  = float(len(passwords))
    unique = float(len(set(passwords)))
    return (total - unique) / total * 100.0

def classify_strength(password):
    has_lower   = bool(re.search(r'[a-z]', password))
    has_upper   = bool(re.search(r'[A-Z]', password))
    has_digit   = bool(re.search(r'\d',    password))
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
    return {k: dist[k] / total * 100.0 for k in ['Faible', 'Moyen', 'Mixte', 'Fort']}

def compute_coverage(gen_set, eval_set):
    if not eval_set:
        return 0.0
    return float(len(gen_set & eval_set)) / float(len(eval_set)) * 100.0

def security_status(value, target, mode='gte'):
    if mode == 'gte':
        return '[OK]' if value >= target else '[!!]'
    if mode == 'lte':
        return '[OK]' if value <= target else '[!!]'
    return '[??]'

print(">>> Chargement eval.txt ...")
eval_lines = load_file('eval.txt')
eval_set   = set(eval_lines)
print(">>> eval.txt : {} mots de passe".format(len(eval_set)))

RUNS_DIR = 'runs_gru'
run_files = [
    ('10k.txt',  '10k'),
    ('100k.txt', '100k'),
]

print("\n" + "="*70)
print("EVALUATION GRU vs EVAL.TXT")
print("="*70)

for filename, label in run_files:
    filepath = os.path.join(RUNS_DIR, filename)
    if not os.path.exists(filepath):
        print("\n[{}] FICHIER INTROUVABLE : {}".format(label, filepath))
        continue

    print("\n[{}] Chargement {} ...".format(label, filepath))
    gen_lines = load_file(filepath)
    gen_set   = set(gen_lines)

    ttr      = float(len(gen_set)) / float(len(gen_lines)) if gen_lines else 0.0
    avg_len  = mean([len(p) for p in gen_lines])
    hits     = eval_set & gen_set
    coverage = compute_coverage(gen_set, eval_set)

    entropy       = compute_entropy(gen_lines)
    dup_rate      = compute_duplicate_rate(gen_lines)
    strength_dist = compute_strength_distribution(gen_lines)

    print("")
    print("  -- Metriques generales ------------------------------------------")
    print("  Mots de passe generes (total) : {:,}".format(len(gen_lines)))
    print("  Mots de passe uniques         : {:,}".format(len(gen_set)))
    print("  TTR                           : {:.4f}".format(ttr))
    print("  Longueur moyenne              : {:.2f} chars".format(avg_len))
    print("  Hits sur eval                 : {}".format(len(hits)))
    print("  Couverture eval               : {:.2f}%".format(coverage))

    print("")
    print("  -- Metriques de securite (4.3.4) --------------------------------")
    print("  {:<6} Entropie reelle        : {:.2f} bits  (cible >= 40 bits)".format(
        security_status(entropy, 40, 'gte'), entropy))
    print("  {:<6} Taux de doublons       : {:.2f}%      (cible < 5%)".format(
        security_status(dup_rate, 5, 'lte'), dup_rate))

    print("")
    print("  Force des mots de passe (distribution) :")
    for level in ['Faible', 'Moyen', 'Mixte', 'Fort']:
        pct    = strength_dist.get(level, 0.0)
        bar    = '#' * int(pct / 2)
        marker = ' <-- CIBLE' if level == 'Mixte' else ''
        print("    {:6s} : {:5.1f}%  {}{}".format(level, pct, bar, marker))

    mixte_pct = strength_dist.get('Mixte', 0.0) + strength_dist.get('Fort', 0.0)
    print("")
    print("  {:<6} Mots de passe Mixte+Fort : {:.1f}%  (cible : Mixte dominant)".format(
        '[OK]' if mixte_pct >= 30 else '[!!]', mixte_pct))
    print("  {:<6} Couverture eval set       : {:.2f}% (cible : Maximale)".format(
        '[OK]' if coverage > 0 else '[!!]', coverage))

    if hits:
        print("")
        print("  Exemples de hits :", list(hits)[:10])

print("\n" + "="*70)
print("DONE -- evaluation GRU terminee")
print("="*70)
print(">>> Log sauvegarde dans : {}".format(LOG_FILE))