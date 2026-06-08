# L'Arène des Algos — Jour 1

Projet de **Mohammed MOSLEH** — premier pipeline de Machine Learning de bout en bout.

Le but : prendre un vrai dataset, le faire passer par le pipeline complet
(charger → séparer → entraîner → prédire → mesurer), puis faire s'affronter
plusieurs algos sur un même classement — l'Arène.

## Le problème

Classification supervisée sur deux jeux de données de scikit-learn :

- **Cancer du sein** (`load_breast_cancer`) : 569 patients, 30 mesures, tumeur bénigne ou maligne (2 classes).
- **Vin** (`load_wine`) : 178 échantillons, 13 mesures, 3 classes (donc du multi-classe).

## Les algos comparés

- Régression logistique
- KNN (k plus proches voisins)
- Arbre de décision

Plus un détour non-supervisé avec **KMeans** (Phase 4).

## Le classement (résumé)

**Cancer du sein** (données mises à l'échelle) :

| Rang | Algo | Accuracy |
|------|------|----------|
| 1 | Régression logistique | 98.2 % |
| 1 | KNN | 98.2 % |
| 3 | Arbre de décision | 91.2 % |

**Vin** :

| Rang | Algo | Accuracy |
|------|------|----------|
| 1 | Régression logistique | 94.4 % |
| 1 | Arbre de décision | 94.4 % |
| 3 | KNN | 75.0 % |

Leçon : **aucun algo n'est roi partout**. Le podium change selon le dataset.

## Mon champion : la régression logistique

Une fois les données mises à l'échelle, elle est en tête sur les deux datasets. Mais
je ne la choisis pas que pour l'accuracy :

- **Explicable** : on peut dire quelles mesures pèsent dans la décision. En santé
  (comme en banque ou assurance), pouvoir justifier une décision compte autant que
  la performance.
- **Rapide** à entraîner.
- Sur le cancer du sein, l'erreur qui fait vraiment peur c'est de **rater une tumeur
  maligne**. Je préfère donc un modèle que je peux analyser à une boîte noire un
  poil plus performante.

## Ce que j'ai appris en chemin

- **Scaling** : KNN et la régression logistique gagnent beaucoup à la mise à l'échelle
  (ils raisonnent par distances / poids) ; l'arbre s'en fiche (il découpe par seuils).
- **Fuite de données** : on ajuste **toujours** le `StandardScaler` sur le train seul.
  Laisser le scaler « voir » le test gonfle l'accuracy de façon mensongère.
- L'**API uniforme** de scikit-learn (`.fit` / `.predict`) permet de changer d'algo ou
  de dataset sans réécrire le pipeline.

## Contenu du repo

- `arene_des_algos.ipynb` — le notebook complet, 8 phases, résultats et graphiques affichés.
- `README.md` — ce fichier.

## Lancer le notebook

```bash
pip install scikit-learn pandas numpy matplotlib jupyter
jupyter notebook arene_des_algos.ipynb
```
