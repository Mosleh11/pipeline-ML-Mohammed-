# Jour 3 — L'Arène des Algos : le Fight des IA

Projet de **Mohammed MOSLEH**. Quatre problèmes réels, chacun d'une famille différente,
puis le grand combat : tous les algos s'affrontent sur le même jeu, même découpage, même
métrique, et on sort un leaderboard.

Notebook : [fight_des_ia.ipynb](fight_des_ia.ipynb)

## Les phases

| Phase | Problème | Famille | Datasets |
|-------|----------|---------|----------|
| A | Prix immobiliers Californie | Régression | `fetch_california_housing` (sklearn) |
| B | Segmenter des annonces Airbnb | Clustering (non supervisé) | `airbnb_listings.csv` (Inside Airbnb, Paris) |
| C | SMS spam vs normal | Classification de texte | `sms_spam.tsv` (UCI) |
| D | Signal sonar : mine ou rocher ? | Classification binaire | `sonar.csv` (UCI) |
| E | Le Fight des IA | Leaderboard | sur le sonar |

## Résultats clés

**Phase A — Régression**

| Modèle | R² | MAE | RMSE |
|--------|----|----|------|
| LinearRegression | 0.58 | 0.53 | 0.75 |
| RandomForest | 0.80 | 0.33 | 0.51 |

Le Random Forest écrase la baseline linéaire, mais la linéaire reste lisible et auditable.

**Phase B — Clustering** : 2 segments ressortent (meilleure silhouette). Le scaling est
obligatoire avant KMeans, et un seul outlier (annonce à 100 000/nuit) suffit à casser les clusters.

**Phase C — Spam** : Naive Bayes et régression logistique attrapent l'essentiel des spams
(recall spam > 0.85). On juge sur precision/recall/f1, pas sur l'accuracy (classes déséquilibrées).

**Phase E — Le Fight (sonar, F1)**

| Rang | Algo | F1 | Temps |
|------|------|----|-------|
| 1 | RandomForest | 0.83 | 0.14s |
| 2 | GradientBoosting | 0.83 | 0.14s |
| 3 | SVC rbf | 0.81 | 0.00s |
| 4 | LogisticRegression | 0.80 | 0.00s |
| 5 | DecisionTree | 0.71 | 0.00s |

## Mon champion (justifié au-delà du score)

Le podium **change selon la métrique** :
- en **F1 / accuracy** : RandomForest et GradientBoosting en tête, SVM rbf juste derrière ;
- en **recall** (repérer un maximum de mines) : c'est la **régression logistique** qui gagne.

Sur ce problème, **rater une mine est dramatique** → je privilégie le **recall**, donc la
**régression logistique** : elle attrape le plus de mines, tout en étant rapide et explicable.

Et une leçon de tuning : le SVM rbf passe de **0.81 à 0.86 en F1** une fois réglé avec
GridSearchCV (`C=10, gamma=scale`). On ne juge jamais un algo sur un seul réglage.

Le réflexe à graver : **même split, même métrique adaptée au problème**, et un champion
justifié par le temps, l'interprétabilité et la robustesse — pas juste le score brut.

## Lancer le notebook

```bash
pip install scikit-learn pandas numpy matplotlib jupyter
jupyter notebook fight_des_ia.ipynb
```

Les datasets `sonar.csv`, `sms_spam.tsv` et `airbnb_listings.csv` sont fournis dans le repo
(California Housing est livré avec scikit-learn).
