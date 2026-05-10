# Phase 1 : Analyse du corpus eval.txt + Graphiques
from collections import Counter
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────
path = '../Data/eval.txt'
with open(path, 'r', errors='ignore') as f:
    pwds = [l.strip() for l in f if l.strip()]

print("Total: {} mots de passe".format(len(pwds)))

lengths = [len(p) for p in pwds]
avg_len = sum(lengths) / len(lengths)
print("Longueur moyenne: {:.2f}".format(avg_len))

# ─────────────────────────────────────────────────────────────────────────────
# Graphique 1 : Distribution des longueurs
# ─────────────────────────────────────────────────────────────────────────────
dist = Counter(lengths)
sizes = sorted(dist.keys())

# Longueur la plus fréquente (pour la mettre en évidence)
most_common_len = dist.most_common(1)[0][0]

sizes_20  = [s for s in sizes if s <= 20]
counts_20 = [dist[s] for s in sizes_20]

fig, ax = plt.subplots(figsize=(12, 6))

colors = ['#ff7f0e' if s == most_common_len else '#1f77b4' for s in sizes_20]
bars = ax.bar(sizes_20, counts_20, color=colors, edgecolor='black', linewidth=0.5)

for bar, count in zip(bars, counts_20):
    if count > max(counts_20) * 0.05:  # Annoter seulement les barres significatives
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_20)*0.005,
                '{:,}'.format(count), ha='center', va='bottom', fontsize=8)

ax.set_title('Distribution des longueurs — Corpus Eval', fontsize=14, fontweight='bold')
ax.set_xlabel('Taille du mot de passe (nombre de caractères)', fontsize=12)
ax.set_ylabel('Nombre de mots de passe', fontsize=12)
ax.set_xticks(sizes_20)

legend_elements = [
    Patch(facecolor='#ff7f0e', label='Longueur la plus fréquente ({} chars)'.format(most_common_len)),
    Patch(facecolor='#1f77b4', label='Autres longueurs')
]
ax.legend(handles=legend_elements, fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('graph_eval_distribution_longueurs.png', dpi=150)
plt.close()
print("Graphique sauvegardé -> graph_eval_distribution_longueurs.png")

# ─────────────────────────────────────────────────────────────────────────────
# Graphique 2 : Répartition des catégories de caractères
# ─────────────────────────────────────────────────────────────────────────────
def get_type(p):
    has_l = bool(re.search(r'[a-zA-Z]', p))
    has_n = bool(re.search(r'[0-9]', p))
    has_s = bool(re.search(r'[^a-zA-Z0-9]', p))
    if has_l and has_n and has_s: return 'L + C + Speciaux'
    if has_l and has_n:           return 'Lettres + Chiffres'
    if has_l and has_s:           return 'Lettres + Speciaux'
    if has_n and has_s:           return 'Chiffres + Speciaux'
    if has_l:                     return 'Lettres seules'
    if has_n:                     return 'Chiffres seuls'
    if has_s:                     return 'Speciaux seuls'
    return 'Autre'

types = Counter(get_type(p) for p in pwds)

ordre = [
    'Lettres + Chiffres',
    'Lettres seules',
    'Chiffres seuls',
    'L + C + Speciaux',
    'Lettres + Speciaux',
    'Chiffres + Speciaux',
    'Speciaux seuls',
]
couleurs = ['#1f77b4', '#4e9fd4', '#9ecae1', '#ff7f0e', '#ffbb78', '#d62728', '#aec7e8']

labels  = [o for o in ordre if o in types]
vals    = [types[o] for o in labels]
cols    = [couleurs[i] for i, o in enumerate(ordre) if o in types]
total   = len(pwds)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(labels, vals, color=cols, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, vals):
    ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
            '{:,} ({:.1f}%)'.format(val, val/total*100),
            va='center', fontsize=9)

ax.set_title('Composition des caractères — Corpus Eval', fontsize=14, fontweight='bold')
ax.set_xlabel('Nombre de mots de passe', fontsize=12)
ax.set_xlim(0, max(vals) * 1.25)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('graph_eval_composition_caracteres.png', dpi=150)
plt.close()
print("Graphique sauvegardé -> graph_eval_composition_caracteres.png")

# ─────────────────────────────────────────────────────────────────────────────
# Stats texte + sauvegarde
# ─────────────────────────────────────────────────────────────────────────────
unique  = len(set(pwds))
doublons = len(pwds) - unique

print("\n" + "="*50)
print("STATS GLOBALES — EVAL")
print("="*50)
print("Total            :", len(pwds))
print("Longueur moyenne : {:.2f}".format(avg_len))
print("Min / Max        :", min(lengths), "/", max(lengths))
print("Uniques          :", unique)
print("Doublons         :", doublons)
print("\nCOMPOSITION:")
for t, c in types.most_common():
    print("  {:25s} : {:6d}  ({:.1f}%)".format(t, c, c/total*100))

with open('resultats_eval_phase1.txt', 'w') as out:
    out.write("STATS GLOBALES — EVAL\n")
    out.write("="*50 + "\n")
    out.write("Total            : {}\n".format(len(pwds)))
    out.write("Longueur moyenne : {:.2f}\n".format(avg_len))
    out.write("Min / Max        : {} / {}\n".format(min(lengths), max(lengths)))
    out.write("Uniques          : {}\n".format(unique))
    out.write("Doublons         : {}\n".format(doublons))
    out.write("\nCOMPOSITION:\n")
    for t, c in types.most_common():
        out.write("  {:25s} : {:6d}  ({:.1f}%)\n".format(t, c, c/total*100))

print("Résultats sauvegardés -> resultats_eval_phase1.txt")