# Jour 2 — Nettoyage de données : Telco Customer Churn

Projet de **Mohammed MOSLEH**. On prend un dataset client sale et réaliste et on le mène
de l'état « brut et inexploitable » à l'état « propre, prêt à modéliser », de bout en bout.

Notebook : [nettoyage_telco_churn.ipynb](nettoyage_telco_churn.ipynb)
Dataset : `WA_Fn-UseC_-Telco-Customer-Churn.csv` (~7000 clients, 21 colonnes, cible = `Churn` oui/non).

## Les 8 phases

| Phase | Ce qu'on fait |
|-------|---------------|
| 0 | Charger le CSV brut |
| 1 | Audit qualité : forme, manquants, équilibre de la cible (73/27) |
| 2 | La colonne piégée : `TotalCharges` en texte → 11 trous cachés démasqués et imputés |
| 3 | Encodage : Yes/No → 0/1, nominales → One-Hot, `customerID` jeté |
| 4 | Outliers : règle IQR + boxplots, décision documentée par colonne |
| 5 | Multicolinéarité : heatmap + VIF, suppression de `TotalCharges` |
| 6 | Variables discriminantes : corrélation à la cible + importance Random Forest |
| 7 | Split / scaling propre + démonstration de la fuite de données |
| 8 | Bilan : tableau de bord, comparatif, PCA exploratoire |

## Mes choix de nettoyage (justifiés)

- **TotalCharges** : 11 lignes contenaient un espace au lieu d'un nombre (clients tout
  nouveaux, `tenure=0`). Je les **impute par la médiane** plutôt que de supprimer 11
  vrais clients — la médiane résiste aux extrêmes.
- **Contract** (3 modalités) : encodé en **One-Hot** (nominal) plutôt qu'ordinal. Le lien
  durée d'engagement → churn n'est pas forcément linéaire, on laisse le modèle apprendre.
- **customerID** : supprimé. Un identifiant unique n'est pas une feature — l'encoder en
  One-Hot créerait ~7043 colonnes (explosion de dimensions) et serait une fuite déguisée.
- **Outliers** (tenure, MonthlyCharges, TotalCharges) : **gardés**. Une grosse facture
  n'est pas une erreur, c'est souvent le client le plus rentable (ou celui qui part le
  plus fort). Les supprimer effacerait du signal.
- **Multicolinéarité** : `TotalCharges` ≈ `tenure × MonthlyCharges`. VIF de 8 → on la
  supprime, et le VIF des deux autres redescend sous 5.

## Ce que ça donne

- 21 colonnes → 41 features numériques après One-Hot, **0 trou**, **0 colonne au VIF > 5**,
  7043 lignes conservées.
- Features les plus prédictives du churn : **Contract, tenure, InternetService** → les
  clients en contrat mensuel et récents résilient le plus (histoire métier exploitable
  pour une campagne de rétention).
- **Piège de l'accuracy** : sur du 73/27, répondre toujours « Non » fait déjà 73,5 %.
  Notre régression logistique propre monte à 79,7 % mais surtout récupère 55 % des vrais
  churners (recall) — c'est la vraie métrique à suivre (avant-goût du J4).
- **Fuite de données** : démontrée chiffres en main. Ici l'écart honnête/triche est nul
  (7000 lignes, train et test ont quasi la même moyenne), mais le réflexe reste absolu :
  **on ajuste toujours le scaler sur le train seul**.

Le dataset final est propre et prêt pour l'Arène des algos du Jour 3.

## Lancer le notebook

```bash
pip install scikit-learn pandas numpy matplotlib seaborn statsmodels jupyter
jupyter notebook nettoyage_telco_churn.ipynb
```
