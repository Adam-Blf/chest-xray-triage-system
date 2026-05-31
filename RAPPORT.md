# Rapport · Système d'aide au tri radiologique

**EFREI · M1 Mastère Data Engineering & IA · 2025-2026**
**Trinôme** · Adam Beloucif · Emilien Morice · Arnaud Dissongo

> Squelette suivant à la lettre la structure imposée §5 du sujet *Projet
> Deep Learning – Système d'aide au tri radiologique*. À remplir avec les
> chiffres réels une fois les runs MLflow finalisés (`mlflow ui`).

---

## 1. Problème

**Contexte** · le tri radiologique en première lecture (urgences, dépistage,
téléradiologie) est un goulet d'étranglement · le volume d'images dépasse
la bande passante humaine, ce qui retarde la prise en charge. Un système
d'aide à la décision capable de pré-classer les radiographies thoraciques
par probabilité de pathologie, de **flaguer les cas atypiques** (hors
distribution) et d'**exploiter le compte-rendu** quand il existe libère du
temps médecin sur les cas standards et concentre l'attention sur les cas à
risque.

**Objectif métier** · automatiser la file de tri en sortie de console
acquisition · score multi-label (14 pathologies) · score d'atypicité ·
optionnellement croisement avec un texte de finding pour réduire les
faux positifs.

**Problématique IA** · combiner classification multi-label déséquilibrée,
détection d'anomalies non supervisée et fusion multimodale image+texte
dans un pipeline reproductible, traçable et exposable.

**Intérêt** · validation du pipeline complet sur ChestMNIST (proxy),
extensibilité à NIH ChestX-ray14, OpenI et MIMIC-CXR pour des contextes
plus réalistes.

## 2. Données

| dataset | usage | structure | limites |
| --- | --- | --- | --- |
| ChestMNIST (MedMNIST) | supervisé + AE/VAE + multimodal proxy | 78 468 train · 11 219 val · 22 433 test · 14 labels multi-hot | downsampling agressif (28/64/128/224 px), perte de contexte radiologique fin |
| NIH ChestX-ray14 | comparateur réaliste, métadonnées | ~112 120 images PNG + CSV (âge, sexe, view, follow-up) | labels issus de NLP des reports, bruités |
| Open-i (NLM) | multimodal image + texte vrai | ~7 470 études · XML FINDINGS + IMPRESSION | distribution biaisée centre Indiana |
| MIMIC-CXR-JPG | scaling avancé | ~377 110 images + reports | credentialed PhysioNet, ~480 GB |

**Justification du choix multimodal** · OpenI est l'unique dataset
publiquement téléchargeable de notre liste à offrir des couples
**image + compte-rendu** exploitables. MIMIC-CXR est plus riche mais
demande un agreement PhysioNet (CITI training inclus). NIH ChestX-ray14
fournit seulement les labels issus du NLP sur les reports, pas les
reports eux-mêmes (§3.2 du sujet).

## 3. Analyse exploratoire

Figures et stats produites par `python -m scripts.eda` (dans
`artifacts/figures/`).

### 3.1 Distribution des labels (train, 78 468 images)

| pathologie | support | prévalence |
| --- | --- | --- |
| infiltration | 13 914 | 17.73 % |
| effusion | 9 261 | 11.80 % |
| atelectasis | 7 996 | 10.19 % |
| nodule | 4 375 | 5.58 % |
| mass | 3 988 | 5.08 % |
| pneumothorax | 3 705 | 4.72 % |
| consolidation | 3 263 | 4.16 % |
| pleural_thickening | 2 279 | 2.90 % |
| cardiomegaly | 1 950 | 2.49 % |
| emphysema | 1 799 | 2.29 % |
| edema | 1 690 | 2.15 % |
| fibrosis | 1 158 | 1.48 % |
| pneumonia | 978 | 1.25 % |
| hernia | 144 | 0.18 % |

Conséquences directes · forte sensibilité aux faux négatifs sur les
classes rares (`hernia`, `pneumonia`, `fibrosis`). Le `pos_weight` par
classe dans la BCE compense partiellement, mais l'AUROC macro restera
tirée vers le bas par ces classes.

### 3.2 Co-occurrence

La matrice `P(col | ligne)` (`figures/cooccurrence.png`) confirme les
corrélations cliniques attendues · `effusion` ↔ `atelectasis`,
`infiltration` ↔ `consolidation`, `pneumothorax` rarement seul. Cela
légitime l'usage d'une activation **sigmoïde par classe** + BCE plutôt
que d'une softmax.

### 3.3 Exemples visuels

`figures/examples_positive.png` montre le premier positif de chaque
classe en train. La résolution 64×64 perd les détails fins (nodules de
petite taille) · pour les classes rares, viser la résolution 128×128 ou
224×224 améliore le contraste utile au réseau.

## 4. Préparation

- preprocessing images · resize à 64/128/224 px selon le backbone,
  conversion 1→3 canaux pour les modèles ImageNet
- normalisation · stats ImageNet (mean = `[0.485, 0.456, 0.406]`,
  std = `[0.229, 0.224, 0.225]`)
- augmentation train · `RandomHorizontalFlip(0.5)` + `RandomAffine(±5°, ±2%)`
  pour augmenter la diversité sans casser l'anatomie thoracique
- multi-label · cible float (14,) directement consommée par
  `BCEWithLogitsLoss`
- déséquilibre · `pos_weight = N_neg/N_pos` par classe, clippé à 20
- texte (OpenI) · lower-case + tokenisation interne sur vocabulaire fixé
  à 14 labels + connecteurs (`scripts/openi_loader.py` heuristique avec
  garde-fou de négation)
- stratégie anti-fuite · splits patient-disjoint **fournis par MedMNIST**
  (jamais re-splittés)

## 5. Modélisation supervisée

Trois architectures comparées (`src/models/`) ·

### SimpleCNN (from scratch)

4 blocs conv `(3×3 → BN → ReLU → MaxPool)` · 32 → 64 → 128 → 256 canaux ·
GAP + MLP `(256→128→14)` · dropout 0.3 · 0.5 M paramètres.

### Transfer · ResNet18 / DenseNet121 / EfficientNet-B0

Backbone ImageNet ·  remplacement de la couche FC par `Linear(in, 14)` ·
option `--freeze` (fine-tuning de la tête uniquement). Choix par défaut ·
**ResNet18** (11 M paramètres) pour le ratio perf/coût CPU.

### ViT-Tiny patch16-224 (timm)

Patch embedding 16×16 → 192-d, 12 blocs attention multi-head, classifier
tête `Linear(192, 14)`. 5.7 M paramètres. Justifié par §4.1 (intérêt du
ViT comparé aux CNN sur images médicales).

Loss commun · `BCEWithLogitsLoss(pos_weight=...)`. Optim · AdamW
(`weight_decay=1e-4`) · scheduler cosine · early stopping macro-AUROC.

## 6. Détection d'anomalies

### ConvAutoencoder

Encodeur 4 conv strided (32→64→128→256), latent 64-d, decoder symétrique.
Score = MSE de reconstruction par image. Seuil à `p99` calculé sur les
reconstructions de validation au meilleur epoch · loggé dans MLflow.

### ConvVAE

Architecture similaire + tête `(μ, log σ²)` et reparameterization. Loss
`MSE + β·KL` avec β=1.0. Score `recon + β·KL` par image. Avantage ·
détection d'outliers latents (KL grand) en plus de la reconstruction
défaillante.

**Protocole** · entraînement sur le train ChestMNIST entier (la grande
majorité des images sont "normales-ish") · seuil sur validation ·
analyse des cas dépassant p99 dans le démonstrateur (visualisation
recon vs original).

**Limite clinique** · un score d'anomalie n'est pas un diagnostic ·
il pointe les images dont la **structure** sort de la distribution
apprise. Un patient sain mais imagé sous angle inhabituel peut être
flagué à tort. À utiliser comme aide au triage, pas comme exclusion.

## 7. Modélisation multimodale

`MultimodalFusionModel` avec trois stratégies ·

| stratégie | description | quand utiliser |
| --- | --- | --- |
| **late** | moyenne des logits image-seule + texte-seul | baseline robuste, modalités indépendantes |
| **early** | concat des deux embeddings → MLP partagé | quand on veut une représentation jointe |
| **intermediate** | cross-attention image↔texte | quand alignement fin nécessaire (mention spatiale) |

Loss · moyenne de 3 BCE (image / texte / fused) pour garder les trois
têtes utilisables au démonstrateur. Comparaison sur le **même run
MLflow** · `val_macro_auroc_image_logits`, `_text_logits`, `_fused_logits`.

**Alignement image-texte** · pour OpenI on s'appuie sur le couplage
explicite `<parentImage id="..."/>` du XML. Pour ChestMNIST (mode
fallback), on synthétise une caption depuis les labels positifs
(`"evidence of cardiomegaly and effusion"`) · sert à valider le pipeline
avant disponibilité OpenI.

**Modalités manquantes** · au démonstrateur le texte est optionnel ·
absence → seule la branche image-seule sert.

## 8. Évaluation

Métriques par run (`src/evaluation.py`) ·

- **macro-AUROC** et **micro-AUROC** · sensibilité au déséquilibre
- **AP macro** · qualité de ranking
- **F1 macro** au seuil 0.5
- **par classe** · AUROC, F1, précision, rappel, support

### 8.1 Résultats actuels (runs MLflow)

Premiers résultats remontés via `python -m scripts.compare_runs` ·

| modèle | epochs | métrique principale | valeur |
| --- | --- | --- | --- |
| **CNN scratch** | 1 | test macro-AUROC | **0.6933** |
| **CNN scratch** | 1 | test micro-AUROC | 0.7739 |
| **CNN scratch** | 1 | test macro-F1 @ 0.5 | 0.1376 |
| **AE conv** | 1 | val MSE reconstruction | 0.2119 |
| **VAE** | 1 | val anomaly score (MSE + KL) | 0.4555 |

Le F1 macro reste bas après 1 epoch · normal vu le déséquilibre extrême
(`hernia` à 0.18 % de prévalence) et l'absence de calibration du seuil
par classe. L'AUROC macro à 0.69 dès la première epoch valide le pipeline
supervisé. Les runs ResNet18 et ViT-Tiny sont prévus pour la version
finale (GPU recommandé) · résultats attendus dans la zone 0.78 - 0.83
selon littérature (CheXNet · ~0.84 sur 14-label NIH ChestX-ray14).

## 9. Tracking MLflow

- expérience unique · `chest-xray-triage`
- tags `family` / `variant` / `backbone` / `fusion` pour filtrer
- params · tout `TrainConfig` / `AEConfig` / `MultimodalConfig` +
  `n_params`, `device`, `model_class`
- métriques · `train_loss`, `val_macro_auroc`, `val_micro_auroc`,
  `val_macro_f1`, `epoch_time_sec`, plus la batterie `test_*` à la fin
- artefacts · checkpoint best par run, exposable dans le démonstrateur

Le **meilleur run** retenu sera celui de plus haut `test_macro_auroc` sur
la famille supervisée · screenshot MLflow UI à insérer dans le PDF final.

## 10. Démonstrateur

`app/streamlit_app.py` · trois panneaux ·

1. **Supervisé** · upload PNG/JPG, sélection du backbone (CNN scratch /
   ResNet18 / DenseNet121 / ViT), top-pathologie + table triée + bar chart
2. **Anomalie** · score AE/VAE + seuil p99 + barre de progression
3. **Multimodal** · activé si on saisit un texte de finding

Cohérence stricte avec MLflow · les checkpoints chargés viennent de
`artifacts/<run_name>_best.pt` produits par les scripts d'entraînement.

## 11. Analyse critique

**Robustesse** ·
- ChestMNIST est un proxy down-samplé · les performances absolues ne
  transposent pas directement à NIH/MIMIC.
- Les labels NIH (et donc ChestMNIST hérité) ont un bruit de NLP estimé
  à 10-15 %.

**Généralisation** ·
- Pas de validation externe sur cohorte indépendante (limite scolaire).
- Aucune correction de biais démographique (sexe/âge non utilisés).

**Erreurs fréquentes** ·
- co-occurrence forte effusion/atelectasis → confusion attendue
- pneumothorax sous-représenté → faible rappel attendu sur cette classe

**Coût calculatoire** · CPU = OK pour CNN scratch et AE/VAE @ 64 px en
quelques minutes par epoch. GPU recommandé pour ViT @ 224 px.

**Apport multimodal** · sur OpenI, on attend une amélioration nette du
F1 sur les classes rares (le texte mentionne explicitement la pathologie)
quand l'image est ambiguë. Sur ChestMNIST fallback synthétique, la fusion
**ne peut que dégrader** (le texte est dérivé du label, donc fuite).

**Apport AE/VAE** · scoring atypicité orthogonal à la classification ·
détecte les artefacts (mauvais positionnement, prothèse, sonde) que la
classification multi-label n'a pas appris à juger.

## 12. Conclusion et perspectives

Le repo livre un pipeline reproductible couvrant les six contraintes du
sujet · supervision multi-architecture, anomalies, multimodal, MLflow,
démonstrateur et discipline d'expérimentation. Les performances exactes
à insérer dans la version finale du rapport seront extraites de MLflow
après l'entraînement complet (10 epochs CNN, 10 epochs transfer, 8
epochs ViT, 15 epochs AE/VAE, 8 epochs multimodal).

**Perspectives** ·
- migration vers OpenI réel pour valider la composante multimodale
- pré-traitement spécifique radiographies (CLAHE, normalisation
  segmentale thoracique)
- distillation du ViT vers un CNN compact pour déploiement edge
- comparaison à une baseline CheXNet (DenseNet121 entraîné full)
- intégration explicite des métadonnées NIH (sexe, âge, view position)
- évaluation patient-level (sous-cohorte de validation externe)
