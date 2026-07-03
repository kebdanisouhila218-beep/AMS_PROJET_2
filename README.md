# Password Factory — Evaluation de modeles Deep Learning pour la generation de mots de passe

Projet realise dans le cadre du cours Apprentissage Machine et Statistique (AMS), Master ILSEN (Ingenierie du Logiciel de la Societe Numerique), Avignon Universite.

Realise en binome avec Elyes Chouchane.

Ce travail a donne lieu a la redaction d'un article scientifique : *"Password Factory: Evaluating Deep Learning Architectures for Offensive Password Generation"* (mai 2026), disponible dans ce depot.

## Objectif

Comparer trois architectures de reseaux de neurones profonds — **GRU**, **LSTM** et **Transformer** — sur une tache de generation de dictionnaires de mots de passe destines a l'audit de securite. L'objectif est d'evaluer laquelle de ces architectures modelise le mieux les habitudes humaines de creation de mots de passe, a la fois en termes de couverture (capacite a retrouver de vrais mots de passe) et de qualite de generation (diversite, entropie, fidelite aux structures reelles).

## Structure du depot

```
AMS_PROJET_2/
├── Data/                    Corpus d'entrainement et d'evaluation (train.txt, eval.txt)
├── Phase1_Analyse/          Analyse statistique du corpus (script + graphes + resultats)
└── Phase2/
    ├── GRU/                 Entrainement, generation et evaluation du modele GRU
    ├── LSTM/                Entrainement, generation et evaluation du modele LSTM
    └── Transformer/         Entrainement, generation et evaluation du modele Transformer
```

Chaque dossier de `Phase2` contient un script d'entrainement, un script de generation, un script d'evaluation, le vocabulaire (tokenisation caractere par caractere), et les scripts SLURM utilises pour l'entrainement sur cluster de calcul GPU.

## Corpus

Le corpus d'entrainement est compose de **375 853 entrees** de mots de passe reels.

| Indicateur | Valeur |
|---|---|
| Total d'entrees | 375 853 |
| Entrees uniques | 375 819 |
| Doublons | 34 |
| Longueur moyenne | 7.60 caracteres |
| Longueur min / max | 1 / 110 caracteres |
| Taille du vocabulaire | 93 caracteres |

Le corpus est domine par les mots de passe alphanumeriques (52.2%) et purement alphabetiques (35.0%). Les caracteres speciaux sont tres rares (moins de 0.6% combines), ce qui limite la capacite des modeles a apprendre ce type de motif.

Un desequilibre critique a ete identifie entre les jeux d'entrainement et d'evaluation initiaux (0.2% de caracteres speciaux en entrainement contre 73% en evaluation). Un rebrassage complet du corpus suivi d'une nouvelle repartition (70% / 30%) a permis d'obtenir une distribution homogene entre les deux ensembles (0.50 a 0.52% de caracteres speciaux), condition necessaire pour que les scores d'evaluation refletent une reelle capacite de generalisation plutot qu'un artefact des donnees.

Le detail de cette analyse (scripts, graphes de distribution) se trouve dans `Phase1_Analyse/`.

## Architectures evaluees

Les trois modeles partagent la meme tokenisation au niveau caractere (vocabulaire de 93 tokens), une longueur de sequence de 10, l'optimiseur Adam (taux d'apprentissage 0.001), un arret anticipe (EarlyStopping) sur la perte de validation, et une sauvegarde du meilleur modele (ModelCheckpoint). Les entrainements ont ete executes sur cluster de calcul (GPU Tesla P100, TensorFlow 1.14 / Keras 2.2.4, jobs SLURM).

| Modele | Architecture | Parametres |
|---|---|---|
| GRU | Embedding(64) -> GRU(128) -> GRU(128) -> Dense(93) | 190 749 |
| LSTM | Embedding(64) -> LSTM(128) -> Dropout(0.3) -> Dense(93) | 116 765 |
| Transformer | Embedding(128) + Positional Encoding -> 2x TransformerBlock (4 tetes, d_ff=256) -> Flatten -> Dense(93) | 395 997 |

## Protocoles d'evaluation

Deux protocoles complementaires ont ete concus pour evaluer les modeles au-dela de la seule performance d'entrainement :

**Protocole 1 — Couverture (Hit-Rate)**
Mesure la capacite a retrouver de vrais mots de passe d'un ensemble d'evaluation disjoint de 2000 entrees jamais vues durant l'entrainement, a differentes echelles de generation (10k, 100k, 1M), avec une strategie d'injection de graines mixte (50% corpus d'entrainement, 50% prefixes d'evaluation) et un echantillonnage en temperature pour garantir la diversite des sorties.

**Protocole 2 — Qualite multi-metrique**
La couverture seule etant insuffisante (un modele generant des mots de passe triviaux pourrait obtenir des correspondances non nulles), ce protocole evalue quatre dimensions complementaires : l'entropie reelle (cible : 40 bits ou plus), le taux de doublons (cible : moins de 5%), le ratio type-token (diversite du vocabulaire genere), et la couverture des bigrammes (fidelite aux motifs sequentiels reels).

## Resultats

### Performance d'entrainement

| Metrique | GRU | LSTM | Transformer |
|---|---|---|---|
| Epoques | 14 | 12 | 10 |
| Perte de validation | 2.1094 | 2.0303 | 1.7031 |
| Precision de validation | 49.56% | 47.53% | 56.09% |
| Perplexite de validation | 4.32 | 4.08 | 3.26 |
| Parametres | 190 749 | 116 765 | 395 997 |
| Temps d'entrainement | ~7 500 s | ~5 940 s | ~10 530 s |

### Couverture (Protocole 1)

| Modele | 10k generations | 100k generations | 1M generations |
|---|---|---|---|
| GRU | 0.05% | 0.25% | 1.75% |
| LSTM | 0.25% | 1.15% | 2.05% |
| Transformer | 2.45% | 3.25% | 7.15% |

Le Transformer domine a toutes les echelles, avec une couverture 3.5 fois superieure au LSTM et 28.6 fois superieure au GRU a 1M generations.

### Qualite (Protocole 2, a 1M generations)

| Metrique | GRU | LSTM | Transformer |
|---|---|---|---|
| Entropie | 36.9 bits (sous le seuil) | 38.2 bits (sous le seuil) | 40.5 bits |
| TTR | 0.9999 | 1.0 | 1.0 |
| Couverture bigrammes | 84% | 88.4% | 99.09% |
| Taux de doublons | 0.01% | 0.00% | 0.00% |

Le Transformer est la seule architecture a depasser le seuil d'entropie cible et atteint une couverture de bigrammes proche de la totalite, confirmant sa superiorite dans la modelisation fidele des structures reelles de mots de passe.

## Analyse critique

- **Superiorite du Transformer** : le mecanisme de self-attention permet de capturer le contexte complet d'une sequence de caracteres simultanement, contrairement aux architectures sequentielles (GRU, LSTM) qui traitent l'information de gauche a droite et peuvent perdre le contexte distant.
- **GRU face a LSTM** : malgre un nombre de parametres superieur (190 749 contre 116 765), le GRU est surpasse par le LSTM sur toutes les metriques. Ce resultat suggere que pour cette tache specifique (sequences courtes et heterogenes), le mecanisme de portes plus expressif du LSTM offre un avantage reel par rapport au schema simplifie du GRU.
- **Effet d'echelle** : la couverture du Transformer progresse de maniere quasi lineaire avec le volume de generation, tandis que le GRU atteint un plateau precoce.
- **Limites identifiees** : l'ecart de longueur moyenne entre corpus d'entrainement (7.60 caracteres) et d'evaluation (10.37 caracteres), ainsi que la raret des caracteres speciaux dans le corpus d'entrainement (0.52%), limitent la couverture globale. Ces deux points motivent un travail futur d'augmentation du corpus.
- **Cout d'entrainement** : le Transformer necessite le temps d'entrainement le plus long en raison de son nombre de parametres, mais le GRU, malgre moins de parametres, requiert davantage d'epoques pour converger, annulant son avantage theorique de rapidite. Le LSTM offre le meilleur compromis efficacite/performance parmi les architectures recurrentes.

## Repartition du travail

Projet realise en binome avec Elyes Chouchane : analyse du corpus, implementation et entrainement des trois architectures, conception des protocoles d'evaluation et redaction de l'article ont ete menes conjointement.

## Article

Kebdani, S., Chouchane, E. (2026). *Password Factory: Evaluating Deep Learning Architectures for Offensive Password Generation*. Master Informatique — Ingenierie du Logiciel de la Societe Numerique (ILSEN), Avignon Universite.

## Competences mobilisees

Python, TensorFlow, Keras, Deep Learning, GRU, LSTM, Transformer, NLP, calcul sur cluster GPU (SLURM), analyse statistique de corpus, conception de protocoles d'evaluation, redaction scientifique.
