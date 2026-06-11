# Jour 4 — Évaluation rigoureuse et déploiement

Projet de **Mohammed MOSLEH**. On devient **arbitres** : évaluer plusieurs modèles avec
une rigueur de pro, puis déployer le champion (sauvegarde → API → WebApp).

Dataset : cancer du sein (classification binaire, 569 lignes).

## Fichiers

| Fichier | Rôle |
|---------|------|
| [evaluation_et_deploiement.ipynb](evaluation_et_deploiement.ipynb) | Le notebook complet (phases 1→7, résultats affichés) |
| `modele.joblib` | Le champion sérialisé (modèle + scaler + plages d'entraînement) |
| `api.py` | API Flask : endpoint `POST /predict` |
| `app.py` | WebApp Streamlit de prédiction |

## Les phases

1. **Split train / validation / test** avec `stratify` (341 / 114 / 114) + checkpoints
   (val_size=0, dataset 95/5).
2. **Bootstrap** : score moyen ± écart-type sur 30 rééchantillonnages avec remise (OOB).
3. **Validation croisée k-fold** : moyenne + dispersion fiables, cas leave-one-out et
   StratifiedKFold sur déséquilibré.
4. **Métrique selon le coût métier** : l'accuracy ment sur les données déséquilibrées.
   On compare deux modèles sur le **coût métier** (rater une tumeur maligne = 10, fausse
   alerte = 1), pas sur l'accuracy.
5. **Sérialisation + logique d'API** : modèle + scaler sauvés ensemble, fonction de
   prédiction validée sur 4 cas (normal, mauvais nombre de valeurs, texte, hors-plage).
6. **WebApp** : `app.py` (Streamlit).
7. **Arbitrage final** : leaderboard agrégé (accuracy CV, recall, coût métier, temps
   d'entraînement, latence) — Random Forest vs Gradient Boosting vs **PMC (réseau de
   neurones)**.

## Le verdict

Sur ce dataset **tabulaire**, le PMC (réseau de neurones) est compétitif mais ne creuse
pas d'écart décisif face aux algos classiques. La leçon de la semaine : **sur du tabulaire,
un bon algo classique tient tête à un réseau de neurones, en plus rapide et plus
explicable**.

→ Je déploie la **régression logistique** : explicable, instantanée, et avec un recall
élevé une fois le seuil réglé. En santé, pouvoir justifier une décision compte autant que
le score. C'est elle qui est dans `modele.joblib`.

Le réflexe d'arbitre : on ne juge pas un modèle sur l'accuracy seule, mais sur le **coût
métier**, la **stabilité** (bootstrap, k-fold), le **temps** et l'**explicabilité**.

## Lancer le projet

```bash
# 1. Le notebook d'évaluation (génère modele.joblib)
pip install scikit-learn pandas numpy matplotlib jupyter joblib
jupyter notebook evaluation_et_deploiement.ipynb

# 2. L'API (machines)
pip install flask
python api.py
# puis : curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\": [...30 nombres...]}"

# 3. La WebApp (humains)
pip install streamlit
streamlit run app.py
```

> Note : le PMC utilise ici le `MLPClassifier` de scikit-learn (un vrai réseau de neurones,
> sans dépendance GPU). Keras/TensorFlow ferait l'équivalent avec `model.save("modele.keras")`.
